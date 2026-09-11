"""求解后端：HiGHS 经 Pyomo 的 LP 建模与求解。

规范要求（第 4 节第 6—8 条）：
  * 对外返回求解状态、是否有可行解、目标、耗时与最大残差，不只返回一个数组；
  * 不把所有异常捕获后返回零数组；区分输入错误、不可行、无界与超时；
  * 只有经过复算的可行解才能继续执行。

实现说明
--------
HiGHS 通过 Pyomo 有两条接口，两者对"不可行"和"取解"的行为不同：

1. ``appsi_highs``（推荐）：可行时用 ``results.solution_loader()`` 把解写回变量，
   不可行时不抛异常而是通过 ``found_feasible_solution()`` 返回 False。
2. ``highs``（legacy SolverFactory 路径）：是备选后端。

本模块优先用 1；若该后端不可用则退回 2。**任何取解失败都判为失败并返回
不可行报告，绝不允许把变量初值当成最优解。**

结构复用（2026 版新增）
------------------------
``solve_model`` 接受一个可选的既有求解器实例 ``solver``。同一个 Pyomo 模型
对象 + 同一个 appsi 实例时，``opt.solve()`` 走 appsi 的 ``update()`` 增量路径
（只把改动的参数、约束边界、目标系数推给 HiGHS），而不是 ``set_instance()``
重新翻译整个模型；这正是滚动窗口逐次重解时的主要开销来源。
默认 ``solver=None`` 的行为与改动前完全一致。
"""

from __future__ import annotations

import time
import warnings

import pyomo.environ as pyo

from .schemas import SolveReport

#: 默认求解器名
DEFAULT_SOLVER = "appsi_highs"

_AVAILABLE: dict[str, bool] = {}


def solver_available(name: str = DEFAULT_SOLVER) -> bool:
    if name not in _AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pyo.SolverFactory(name).available(exception_flag=False)
            _AVAILABLE[name] = True
        except Exception:
            _AVAILABLE[name] = False
    return _AVAILABLE[name]


def _fail(problem: str, solver_name: str, wall: float, message: str) -> SolveReport:
    return SolveReport(
        problem=problem,
        status="error",
        feasible=False,
        objective_yuan=None,
        termination="error",
        solver=solver_name,
        wall_seconds=wall,
        message=message,
    )


def make_appsi_solver(solver_name: str = DEFAULT_SOLVER):
    """取得一个可复用的原生 appsi 求解器实例（非 appsi 名称返回 None）。

    复用同一个求解器实例的意义：appsi 的 ``solve(model)`` 会先比较
    ``model is self._model``——相同则走 ``update()`` 增量路径，只把改动的
    参数/边界推给 HiGHS；不同则 ``set_instance()`` 把整个模型重新翻译一遍。
    结构复用的场景（同一个 Pyomo 模型只改参数取值）因此必须复用实例。
    """
    return _appsi_solver(solver_name)


def solve_model(
    model: pyo.ConcreteModel,
    problem: str,
    *,
    solver_name: str = DEFAULT_SOLVER,
    tee: bool = False,
    time_limit_s: float | None = None,
    mip_gap: float | None = None,
    solver: object | None = None,
) -> SolveReport:
    """求解并返回 :class:`SolveReport`。不吞异常、不伪造成功。

    ``solver``
        可选的既有求解器实例。传入与 ``model`` 绑定的同一实例时，appsi 只做
        增量更新，不再把模型重新传给 HiGHS；传 None（默认）时行为与改动前
        完全一致——每次新建实例、整模型传输。
    """
    if not solver_available(solver_name):
        raise RuntimeError(
            f"求解器 {solver_name} 不可用。请先安装 highspy（pip install highspy）。"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        if solver_name.startswith("appsi"):
            report = _solve_appsi(
                model, problem, solver_name, tee, time_limit_s, mip_gap, t0, solver
            )
        else:
            report = _solve_legacy(
                model, problem, solver_name, tee, time_limit_s, mip_gap, t0
            )

    if report.feasible:
        # 优先用求解器给出的最优目标值，其次从模型的标准名 obj 读取（我们的模型统一叫 obj）。
        objective: float | None = None
        try:
            obj_cmp = model.component("obj")
            if obj_cmp is not None and obj_cmp.active:
                objective = float(pyo.value(obj_cmp))
        except Exception:
            objective = None
        if objective is None:
            try:
                objectives = list(model.component_data_objects(pyo.Objective, active=True))
                if objectives:
                    objective = float(pyo.value(objectives[0]))
            except Exception:
                objective = None
        if objective is None:
            return _fail(problem, solver_name, report.wall_seconds, "无法读取目标值")
        report.objective_yuan = objective
    return report


# --------------------------------------------------------------------------
# appsi 后端
# --------------------------------------------------------------------------


def _appsi_solver(solver_name: str):
    """取得 **原生** appsi 求解器实例（不走 SolverFactory 的 legacy 包装）。

    为什么必须这样：``pyo.SolverFactory('appsi_highs')`` 返回的是
    ``LegacySolver`` 包装器，它的 ``solve()`` 每次都会
    ``self.config = self.config()`` 重建配置并把 ``load_solution`` 覆盖成
    ``load_solutions`` 参数的值，因此外部设置的 ``config.load_solution=False``
    会被静默忽略；不可行时它还会直接抛 RuntimeError。
    直接用原生实例才能：① 关闭自动加载；② 从 ``_last_results_object`` 拿到
    ``solution_loader`` 与真实终止条件。
    """
    if solver_name.startswith("appsi"):
        try:
            from pyomo.contrib.appsi.solvers import Highs  # type: ignore

            return Highs()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"无法创建原生 appsi Highs 求解器：{exc}") from exc
    return None


def _reset_solver_state(opt: object) -> bool:
    """丢掉 HiGHS 上一轮求解留下的基/解状态，使下一次 ``run()`` 是冷启动。

    为什么复用实例必须做这一步：模型对象不变时 appsi 走 ``update()`` 增量路径，
    HiGHS 会拿上一次的单纯形基做热启动；在退化的 LP 上热启动可能收敛到另一个
    最优顶点，解向量会差几个 ULP。滚动仿真要求"复用结构"与"每次重建"逐位一致
    （每次重建时 HiGHS 本来也是冷启动），所以复用路径强制冷启动。
    这不会削弱求解：模型、变量、约束、目标、边界一字未改，只是丢弃基。

    返回 False 表示该 highspy 版本没有 ``clearSolver``，无法保证冷启动。
    """
    model = getattr(opt, "_solver_model", None)
    if model is None:
        # 还没绑定任何模型，本来就没有上一轮的基要清。
        return True
    clear = getattr(model, "clearSolver", None)
    if clear is None:
        return False
    clear()
    return True


def _solve_appsi(
    model: pyo.ConcreteModel,
    problem: str,
    solver_name: str,
    tee: bool,
    time_limit_s: float | None,
    mip_gap: float | None,
    t0: float,
    solver: object | None = None,
) -> SolveReport:
    opt = solver if solver is not None else _appsi_solver(solver_name)
    if opt is None:
        return _fail(problem, solver_name, time.perf_counter() - t0, "非 appsi 求解器名")
    if solver is not None and not _reset_solver_state(opt):
        # 清不掉上一轮的基就不能保证与冷启动一致，退回"每次新建实例"的老路子。
        opt = _appsi_solver(solver_name)
    # 关键：不要自动加载。不可行时原生接口不会抛异常，而是给出终止条件。
    opt.config.load_solution = False
    if tee:
        opt.config.stream_solver = True
    if time_limit_s is not None:
        try:
            opt.config.time_limit = float(time_limit_s)
        except Exception:
            pass
    if mip_gap is not None:
        try:
            opt.config.mip_gap = float(mip_gap)
        except Exception:
            pass

    try:
        opt.solve(model)
    except Exception as exc:
        return _fail(problem, solver_name, time.perf_counter() - t0, f"求解异常：{exc}")

    wall = time.perf_counter() - t0
    results = getattr(opt, "_last_results_object", None)
    if results is None:
        return _fail(problem, solver_name, wall, "求解器未返回结果对象 (_last_results_object 为空)")

    termination = str(getattr(results, "termination_condition", "unknown"))
    loader = getattr(results, "solution_loader", None)

    if loader is None:
        # 不可行 / 无界 / 超时无解：如实报告终止条件，不加载、不伪造
        return SolveReport(
            problem=problem,
            status=termination.lower(),
            feasible=False,
            objective_yuan=None,
            termination=termination.lower(),
            solver=solver_name,
            wall_seconds=wall,
            message=f"未找到可行解，终止条件：{termination}",
        )

    try:
        if callable(loader):
            loader()          # legacy 风格的加载器
        else:
            loader.load_vars()  # appsi 的 PersistentSolutionLoader
    except Exception as exc:
        return _fail(problem, solver_name, wall, f"解加载失败：{exc}")

    # 载入后复核：必须存在被赋值的变量，避免"零解冒充最优解"
    assigned = sum(
        1 for var in model.component_data_objects(pyo.Var, active=True) if var.value is not None
    )
    if assigned == 0:
        return _fail(problem, solver_name, wall, "解加载后仍无任何变量被赋值")

    return SolveReport(
        problem=problem,
        status="ok",
        feasible=True,
        objective_yuan=None,
        termination=termination.lower(),
        solver=solver_name,
        wall_seconds=wall,
    )


# --------------------------------------------------------------------------
# legacy SolverFactory 后端（备选）
# --------------------------------------------------------------------------


def _solve_legacy(
    model: pyo.ConcreteModel,
    problem: str,
    solver_name: str,
    tee: bool,
    time_limit_s: float | None,
    mip_gap: float | None,
    t0: float,
) -> SolveReport:
    opt = pyo.SolverFactory(solver_name)
    options: dict[str, object] = {}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    if mip_gap is not None:
        options["mip_gap"] = float(mip_gap)

    try:
        results = opt.solve(model, tee=tee, load_solutions=True, options=options or None)
    except Exception as exc:
        return _fail(problem, solver_name, time.perf_counter() - t0, f"求解异常：{exc}")

    wall = time.perf_counter() - t0
    status = str(results.solver.status).lower()
    termination = str(results.solver.termination_condition).lower()
    feasible = status in ("ok", "warning") and termination in ("optimal", "feasible")
    return SolveReport(
        problem=problem,
        status=status,
        feasible=feasible,
        objective_yuan=None,
        termination=termination,
        solver=solver_name,
        wall_seconds=wall,
        message="" if feasible else f"未找到可行解：{status}/{termination}",
    )


__all__ = ["solve_model", "solver_available", "make_appsi_solver", "DEFAULT_SOLVER"]
