"""原始附件读取与标准化。

职责边界（规范第 5 节"五个必须隔离的接口"）：
  * 只负责**原始附件 -> 规范化数组**，不许私自补未来预测、不许改题面规则；
  * 原附件与模板一律只读；排除 Excel 锁文件 ``~$*.xlsx``；
  * 时间换算全部委托给 :mod:`microgrid.timeaxis`，本模块不自行算段号。

单位口径（备忘录第 4 节）：
  附件里的 10 分钟数值是**区间平均功率**（kW），
  区间电量 = 功率 / 6（kWh）；价格直接是区间平均单价（元/kWh），不除以 6。

标签口径（2026-09-11 定稿，见 :mod:`microgrid.timeaxis`）：
  源标签是**区间起点**：'00:10' 表示 [00:10, 00:20)，序号 v=0；
  '23:50' 表示 [23:50, 00:00)，序号 v=143；
  '0:00+1' / '24:00' 表示 [次日 00:00, 次日 00:10)，序号 v=143（同列）。

  自然日的首个区间（00:00—00:10）由**上一源日的 '0:00+1' 列**承担，
  这一步在 :mod:`microgrid.timeline` 里做，本模块只吐出源序列。
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from .constants import (
    ATTACHMENT_1,
    ATTACHMENT_2,
    ATTACHMENT_3,
    ATTACHMENT_4,
    DELTA_T_HOURS,
    FORECAST_PUBLISH_HOURS,
    LOCK_PREFIX,
    N_INTERVAL,
)
from .timeaxis import (
    DATE_2025_01_01,
    DATE_2025_12_31,
    calendar_days,
    seq_index_from_label,
    to_date,
)


class DataError(RuntimeError):
    """输入数据不符合预期结构——必须让流程失败，而不是打印警告后继续。"""


# ==========================================================================
# 项目根目录解析（不允许在源码里写个人绝对路径）
# ==========================================================================


def project_root(start: Path | None = None) -> Path:
    """从当前文件向上找含 ``data/`` 与 ``Pr/`` 的目录作为项目根。"""
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data").is_dir() and (candidate / "Pr").is_dir():
            return candidate
    env = os.environ.get("MICROGRID_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    raise DataError("无法定位项目根目录：请设置 MICROGRID_ROOT 或从仓库内运行")


def data_path(relative: str, root: Path | None = None) -> Path:
    p = (root or project_root()) / relative
    if p.name.startswith(LOCK_PREFIX):
        raise DataError(f"拒绝读取 Excel 锁文件：{p}")
    if not p.exists():
        raise DataError(f"数据文件不存在：{p}")
    return p


# ==========================================================================
# 通用工作表读取
# ==========================================================================


def _sheet_rows(path: Path, sheet: str | int = 0):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
        for row in ws.iter_rows(values_only=True):
            yield row
    finally:
        wb.close()


def _grid_from_matrix_sheet(path: Path, sheet: str) -> tuple[list[date], np.ndarray]:
    """读取"日期 + 144 个时间列（结束标签）"的矩阵型工作表。

    Returns
    -------
    days : 长度为 365 的日期列表
    values : (365, 144) 数组，列已按模型区间号排序
    """
    rows = list(_sheet_rows(path, sheet))
    if not rows:
        raise DataError(f"{path.name}/{sheet} 为空")
    header = rows[0]
    time_cols: list[tuple[int, int]] = []  # (列下标, 区间号)
    for col, label in enumerate(header[1:], start=1):
        if label is None:
            continue
        if isinstance(label, (time, str)):
            try:
                t = seq_index_from_label(label)
            except ValueError:
                continue
            time_cols.append((col, t))
    if len(time_cols) != N_INTERVAL:
        raise DataError(f"{path.name}/{sheet} 时间列数应为 {N_INTERVAL}，得到 {len(time_cols)}")
    time_cols.sort(key=lambda pair: pair[1])
    if [t for _, t in time_cols] != list(range(N_INTERVAL)):
        raise DataError(f"{path.name}/{sheet} 时间标签无法一一映射到 0..143")

    days: list[date] = []
    values = np.full((len(rows) - 1, N_INTERVAL), np.nan, dtype=np.float64)
    for i, row in enumerate(rows[1:]):
        if row[0] is None:
            continue
        days.append(to_date(row[0]))
        for col, t in time_cols:
            v = row[col]
            if v is None or isinstance(v, str):
                raise DataError(f"{path.name}/{sheet} 第 {i + 2} 行第 {col + 1} 列缺失或非数值：{v!r}")
            values[i, t] = float(v)
    if not days:
        raise DataError(f"{path.name}/{sheet} 没有任何日期行")
    if np.isnan(values).any():
        n_missing = int(np.isnan(values).sum())
        raise DataError(f"{path.name}/{sheet} 存在 {n_missing} 个缺失值")
    return days, values


# ==========================================================================
# 附件1：典型日（电价 / 小区负载 / 光伏发电预测功率）
# ==========================================================================


@dataclass(frozen=True)
class Attachment1:
    price_yuan_per_kwh: np.ndarray   # (144,)
    load_kw: np.ndarray              # (144,)
    pv_forecast_kw: np.ndarray       # (144,)

    @property
    def demand_kwh(self) -> np.ndarray:
        return self.load_kw * DELTA_T_HOURS

    @property
    def pv_kwh(self) -> np.ndarray:
        return self.pv_forecast_kw * DELTA_T_HOURS


def load_attachment1(root: Path | None = None) -> Attachment1:
    path = data_path(ATTACHMENT_1, root)
    rows = list(_sheet_rows(path, 0))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    want = ("时间", "电价", "小区负载", "光伏发电预测功率")
    if header[:4] != list(want):
        raise DataError(f"附件1 表头不符合预期：{header[:4]}，期望 {list(want)}")

    price = np.full(N_INTERVAL, np.nan)
    load = np.full(N_INTERVAL, np.nan)
    pv = np.full(N_INTERVAL, np.nan)
    for row in rows[1:]:
        if row[0] is None:
            continue
        t = seq_index_from_label(row[0])
        for arr, idx in ((price, 1), (load, 2), (pv, 3)):
            if arr[t] == arr[t] and not np.isnan(arr[t]):
                raise DataError(f"附件1 区间 {t} 重复出现")
            arr[t] = float(row[idx])
    for name, arr in (("电价", price), ("小区负载", load), ("光伏预测", pv)):
        if np.isnan(arr).any():
            raise DataError(f"附件1 {name} 缺少区间：{np.flatnonzero(np.isnan(arr)).tolist()}")
    return Attachment1(price_yuan_per_kwh=price, load_kw=load, pv_forecast_kw=pv)


# ==========================================================================
# 附件2：全年小区负载与光伏实际功率
# ==========================================================================


@dataclass(frozen=True)
class Attachment2:
    days: list[date]
    load_kw: np.ndarray              # (365, 144)
    pv_actual_kw: np.ndarray         # (365, 144)

    def index_of(self, day: date) -> int:
        try:
            return self.days.index(day)
        except ValueError as exc:
            raise DataError(f"附件2 不含日期 {day}") from exc

    @property
    def demand_kwh(self) -> np.ndarray:
        return self.load_kw * DELTA_T_HOURS

    @property
    def pv_kwh(self) -> np.ndarray:
        return self.pv_actual_kw * DELTA_T_HOURS


def load_attachment2(root: Path | None = None) -> Attachment2:
    path = data_path(ATTACHMENT_2, root)
    days_l, load = _grid_from_matrix_sheet(path, "小区负载")
    days_p, pv = _grid_from_matrix_sheet(path, "光伏发电实际功率")
    if days_l != days_p:
        raise DataError("附件2 两张工作表的日期不一致")
    return Attachment2(days=days_l, load_kw=load, pv_actual_kw=pv)


# ==========================================================================
# 附件3：光伏发电功率预报（0/6/12/18 发布，未来 24 小时整点平均功率）
# ==========================================================================


@dataclass(frozen=True)
class ForecastBlock:
    """一条发布记录。

    口径（备忘录第 7 节、已冻结）：
      "预报 j 小时" = 发布时刻 a 之后 [a+j-1, a+j) 小时的**平均功率**。
      故 j=1 完整覆盖未来首小时，不需要 0 小时预测、端点补齐或线性插值。
    """

    publish_day: date
    publish_hour: int
    hourly_power_kw: np.ndarray   # (24,) 对应绝对小时 [a, a+1) … [a+23, a+24)

    @property
    def publish_date(self) -> date:
        return self.publish_day


@dataclass(frozen=True)
class Attachment3:
    blocks: tuple[ForecastBlock, ...]

    def blocks_on(self, day: date) -> list[ForecastBlock]:
        out = [b for b in self.blocks if b.publish_day == day]
        out.sort(key=lambda b: b.publish_hour)
        return out

    def block_at(self, day: date, hour: int) -> ForecastBlock:
        for b in self.blocks:
            if b.publish_day == day and b.publish_hour == hour:
                return b
        raise DataError(f"附件3 缺少 {day} {hour}:00 的预报块")

    def hourly_power_on(self, day: date) -> np.ndarray:
        """某日 24 个小时平均功率（kW），取自该日 00:00 的发布。

        00:00 发布的 24 条预报覆盖 [day 00:00, day 24:00)，自足、无需拼接。
        规范要求：不得读取未来 6/12/18 点或次日 0 点的发布来补当前窗口；
        超出单次 24 小时覆盖的部分由**历史光伏点预测**补齐，那一步在
        :mod:`microgrid.forecast` 里做，本方法只负责已发布版本本身。
        """
        return self.block_at(day, 0).hourly_power_kw.copy()

    def hourly_power_from(self, publish_day: date, publish_hour: int) -> np.ndarray:
        """取指定发布版本覆盖 [publish_day 00:00, +24h) 的 24 条小时预报。"""
        return self.block_at(publish_day, publish_hour).hourly_power_kw.copy()

    def latest_published_covering(
        self, before: date, before_hour: int, target_hour_abs: int
    ) -> tuple[ForecastBlock, int] | None:
        """在 (before, before_hour) 之前（含）已发布、且覆盖目标绝对小时的**最新**版本。

        Returns ``(block, offset)``，offset 为目标绝对小时在该块 24 条预报中的
        下标；没有任何版本覆盖时返回 None。绝不读取尚未发布的版本。
        """
        best: tuple[ForecastBlock, int] | None = None
        for block in self.blocks:
            publish_abs = block.publish_day.toordinal() * 24 + block.publish_hour
            request_abs = before.toordinal() * 24 + before_hour
            if publish_abs > request_abs:
                continue
            offset = target_hour_abs - publish_abs
            if 0 <= offset < 24:
                if best is None or (
                    block.publish_day.toordinal() * 24 + block.publish_hour
                    > best[0].publish_day.toordinal() * 24 + best[0].publish_hour
                ):
                    best = (block, offset)
        return best


def load_attachment3(root: Path | None = None) -> Attachment3:
    path = data_path(ATTACHMENT_3, root)
    rows = list(_sheet_rows(path, 0))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    if header[0] != "日期" or header[1] != "预报时刻":
        raise DataError(f"附件3 表头不符合预期：{header[:2]}")
    hour_cols = header[2:]
    if len(hour_cols) != 24 or hour_cols[0] != "预报1小时" or hour_cols[-1] != "预报24小时":
        raise DataError(f"附件3 预报列不符合预期：{len(hour_cols)} 列，首 {hour_cols[:1]} 末 {hour_cols[-1:]}")

    blocks: list[ForecastBlock] = []
    current_day: date | None = None
    seen: set[tuple[date, int]] = set()
    for i, row in enumerate(rows[1:], start=2):
        raw_day, raw_hour = row[0], row[1]
        if raw_day is not None and str(raw_day).strip() != "":
            current_day = to_date(raw_day)
        if current_day is None:
            raise DataError(f"附件3 第 {i} 行在首个日期块之前出现预报行")
        if raw_hour is None:
            raise DataError(f"附件3 第 {i} 行缺少预报时刻")
        hour = _parse_publish_hour(raw_hour)
        values = row[2:26]
        if any(v is None or isinstance(v, str) for v in values):
            raise DataError(f"附件3 第 {i} 行存在缺失预报值")
        key = (current_day, hour)
        if key in seen:
            raise DataError(f"附件3 重复发布块：{current_day} {hour}:00")
        seen.add(key)
        blocks.append(
            ForecastBlock(
                publish_day=current_day,
                publish_hour=hour,
                hourly_power_kw=np.asarray(values, dtype=np.float64),
            )
        )
    if not blocks:
        raise DataError("附件3 未解析出任何预报块")
    return Attachment3(blocks=tuple(blocks))


def _parse_publish_hour(value: object) -> int:
    if isinstance(value, time):
        hour = value.hour
    elif isinstance(value, str):
        text = value.strip()
        if ":" not in text:
            raise DataError(f"无法解析预报时刻：{value!r}")
        hour = int(text.split(":", 1)[0])
    else:
        hour = int(value)  # type: ignore[arg-type]
    if hour not in FORECAST_PUBLISH_HOURS:
        raise DataError(f"预报时刻 {hour} 不在 {FORECAST_PUBLISH_HOURS}")
    return hour


# ==========================================================================
# 附件4：全年实际电价
# ==========================================================================


@dataclass(frozen=True)
class Attachment4:
    days: list[date]
    price_yuan_per_kwh: np.ndarray   # (365, 144)

    def index_of(self, day: date) -> int:
        try:
            return self.days.index(day)
        except ValueError as exc:
            raise DataError(f"附件4 不含日期 {day}") from exc


def load_attachment4(root: Path | None = None) -> Attachment4:
    path = data_path(ATTACHMENT_4, root)
    days, price = _grid_from_matrix_sheet(path, 0)
    return Attachment4(days=days, price_yuan_per_kwh=price)


# ==========================================================================
# 汇总加载
# ==========================================================================


@dataclass(frozen=True)
class DataBundle:
    """一次性加载的全部原始数据（只读）。

    规范第 12 节：只读一次 Excel，之后在内存/缓存里复用，每段不重新打开工作簿。
    """

    attachment1: Attachment1
    attachment2: Attachment2
    attachment3: Attachment3
    attachment4: Attachment4

    def day_index(self, day: date) -> int:
        return self.attachment2.index_of(day)

    def check_calendar(self) -> None:
        expected = calendar_days(DATE_2025_01_01, DATE_2025_12_31)
        for name, days in (
            ("附件2", self.attachment2.days),
            ("附件4", self.attachment4.days),
        ):
            if days != expected:
                raise DataError(f"{name} 日期序列不是 2025-01-01..2025-12-31 连续日历")


@functools.lru_cache(maxsize=4)
def _load_bundle_cached(root_str: str) -> DataBundle:
    root = Path(root_str)
    bundle = DataBundle(
        attachment1=load_attachment1(root),
        attachment2=load_attachment2(root),
        attachment3=load_attachment3(root),
        attachment4=load_attachment4(root),
    )
    bundle.check_calendar()
    return bundle


def load_all(root: Path | None = None) -> DataBundle:
    return _load_bundle_cached(str(root or project_root()))


def input_fingerprint(paths: list[Path]) -> str:
    """输入指纹：路径 + 大小 + mtime，用于检查点/缓存失效判断。"""
    import hashlib

    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        st = p.stat()
        h.update(str(p).encode("utf-8"))
        h.update(str(st.st_size).encode("ascii"))
        h.update(str(int(st.st_mtime)).encode("ascii"))
    return h.hexdigest()[:16]


__all__ = [
    "Attachment1",
    "Attachment2",
    "Attachment3",
    "Attachment4",
    "ForecastBlock",
    "DataBundle",
    "DataError",
    "load_attachment1",
    "load_attachment2",
    "load_attachment3",
    "load_attachment4",
    "load_all",
    "project_root",
    "data_path",
    "input_fingerprint",
]
