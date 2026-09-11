"""时间轴映射：源标签 <-> 模型区间 <-> 输出模板格位。

**本模块是整个工程里唯一允许做时间换算的地方。**
规范第 2 节与本文件共同定义映射；其它模块只许调用这里的函数，
不得自行用字符串切片或浮点小时推算段号（这是 144/145 错位的主要来源）。

三个坐标系
----------
1. 模型区间  t = 0..143，表示 [t*10min, (t+1)*10min)。
2. 模型边界  s = 0..144，表示时刻 s*10min 的瞬时状态。E_0 是 0:00，E_144 是 24:00。
3. 源标签    附件里的 '0:10' … '23:50'、'0:00+1'，全部是**区间结束时刻**。
   '0:10' -> t=0 ；'0:00+1' -> t=143。

4. 输出模板格位（源模板标签整体后移了一段，见下方 TEMPLATE_* 说明）。
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

import numpy as np

from .constants import MINUTES_PER_INTERVAL, N_BOUNDARY, N_INTERVAL

# --------------------------------------------------------------------------
# 源标签解析
# --------------------------------------------------------------------------

_END_OF_DAY_LABEL = "0:00+1"
_TIME_LABEL_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def interval_index_from_label(label: str) -> int:
    """把源时间标签（区间结束时刻）映射为模型区间号。

    >>> interval_index_from_label("0:10")
    0
    >>> interval_index_from_label("0:20")
    1
    >>> interval_index_from_label("0:00+1")
    143
    """
    text = str(label).strip()
    if text == _END_OF_DAY_LABEL:
        return N_INTERVAL - 1
    m = _TIME_LABEL_RE.match(text)
    if not m:
        # openpyxl 有时把 '0:10' 读成 datetime.time
        if isinstance(label, time):
            total = label.hour * 60 + label.minute
            return _minutes_to_interval(total)
        raise ValueError(f"无法解析时间标签：{label!r}")
    hour, minute = int(m.group(1)), int(m.group(2))
    return _minutes_to_interval(hour * 60 + minute)


def _minutes_to_interval(total_minutes: int) -> int:
    # 严格口径：源标签是**区间结束时刻**，只允许 0:10 .. 23:50 与 '0:00+1'。
    # '0:00' / '24:00' 是状态边界标签，不是交易标签，必须显式报错而不是静默取整。
    if total_minutes <= 0 or total_minutes > 24 * 60 - MINUTES_PER_INTERVAL:
        raise ValueError(f"不是合法的区间结束标签：{total_minutes} 分钟")
    if total_minutes % MINUTES_PER_INTERVAL:
        raise ValueError(f"时间标签不是 {MINUTES_PER_INTERVAL} 分钟整数倍：{total_minutes}")
    return total_minutes // MINUTES_PER_INTERVAL - 1


def interval_label_set() -> frozenset[str]:
    """全部合法区间标签（含 '0:00+1'）。"""
    return frozenset(interval_labels())


def interval_labels() -> list[str]:
    """模型区间 0..143 的规范标签（区间结束时刻）。"""
    out = []
    for t in range(N_INTERVAL):
        out.append(_END_OF_DAY_LABEL if t == N_INTERVAL - 1 else _format_minutes((t + 1) * MINUTES_PER_INTERVAL))
    return out


def boundary_labels() -> list[str]:
    """模型边界 0..144 的规范标签。"""
    out = []
    for s in range(N_BOUNDARY):
        if s == N_BOUNDARY - 1:
            out.append(_END_OF_DAY_LABEL)
        else:
            out.append(_format_minutes(s * MINUTES_PER_INTERVAL))
    return out


def _format_minutes(total_minutes: int, plus_one: bool = False) -> str:
    hour, minute = divmod(total_minutes, 60)
    return f"{hour}:{minute:02d}" + ("+1" if plus_one else "")


def boundary_index(interval: int, side: str) -> int:
    """区间 t 的起止边界号。side ∈ {'start', 'end'}。"""
    if not 0 <= interval < N_INTERVAL:
        raise ValueError(f"区间号越界：{interval}")
    return interval if side == "start" else interval + 1


# --------------------------------------------------------------------------
# 输出模板格位
# --------------------------------------------------------------------------
# 实测（只读检查 data/附件5/*.xlsx）：
#   result1 "计划购电量"  A2 = '0:10-0:20' … A145 = '0:00+1-0:10+1'
#   result2/3/4-2/4-3 "计划购电量"/"调整购电量"
#       表头 B1 = '0:10-0:20'，第 1 行共 146 个时间列 + '全天购电量' + '全天购电费'
#     两处模板的时间标签都比模型区间整体后移了一段：缺 '0:00-0:10'，
#     却多出一个落在次日 00:00-00:10 的列。
#   同一批模板的 "充放电量" 表却用标准口径 '0:00-4:00' … '20:00-24:00' + 0:00/24:00 储电量。
#
# 已确认口径（用户决定 2026-09-11）：**模板标签一字不动**，
# 模型第 t 段写入第 t+1 个数据格。即 0:00-0:10 写在 '0:10-0:20' 标签下方，
# 并在交付说明中声明这一映射。
TEMPLATE_SHIFT_INTERVALS = 0

#: 模板每天的时间列数（等于模型区间数，只是标签写法不同）
TEMPLATE_TIME_COLUMNS = N_INTERVAL


def template_column_of_interval(interval: int) -> int:
    """模型区间号 -> 模板时间列序号（0-based，即数据区第几列）。"""
    if not 0 <= interval < N_INTERVAL:
        raise ValueError(f"区间号越界：{interval}")
    return interval + TEMPLATE_SHIFT_INTERVALS


def interval_of_template_column(col: int) -> int:
    """模板时间列序号（0-based） -> 模型区间号。"""
    if not 0 <= col < TEMPLATE_TIME_COLUMNS:
        raise ValueError(f"模板列序号越界：{col}")
    return col - TEMPLATE_SHIFT_INTERVALS


# --------------------------------------------------------------------------
# 日期工具
# --------------------------------------------------------------------------

DATE_2025_01_01 = date(2025, 1, 1)
DATE_2025_12_31 = date(2025, 12, 31)


def calendar_days(start: date = DATE_2025_01_01, end: date = DATE_2025_12_31) -> list[date]:
    n = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n)]


def to_date(value: object) -> date:
    """把 Excel 里的日期单元格规范成 datetime.date。"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip().replace("/", "-")
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"无法解析日期：{value!r}")


# --------------------------------------------------------------------------
# 数值工具
# --------------------------------------------------------------------------


def require_shape(arr: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} 期望形状 {shape}，得到 {out.shape}")
    return out


__all__ = [
    "interval_index_from_label",
    "interval_labels",
    "interval_label_set",
    "boundary_labels",
    "boundary_index",
    "template_column_of_interval",
    "interval_of_template_column",
    "calendar_days",
    "to_date",
    "require_shape",
    "DATE_2025_01_01",
    "DATE_2025_12_31",
    "TEMPLATE_SHIFT_INTERVALS",
    "TEMPLATE_TIME_COLUMNS",
]
