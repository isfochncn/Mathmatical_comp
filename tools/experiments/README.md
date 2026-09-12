# 独立评估与历史诊断

这些脚本各有独立实验用途，不是默认主模型入口。请先读 `Pr/模型优化定稿与验证档案.md`，从项目根目录运行。

- `evaluate_report_forecast.py`：报告校正/融合预测专项，不求解主调度。
- `evaluate_forecast_risk.py`、`evaluate_risk_dispatch.py`：预测误差和同起点短期风险对照；后者必须显式给出 `--baseline-dir`。
- `summarize_forecast_risk.py`：汇总上述实验档案。
- `diagnose_problem3_cost.py`：旧问题二/三二月结果诊断，必须指定 `--problem2-out`、`--problem3-out` 和新 `--out`。
- `replay_problem3_node.py`：旧pooled问题三的固定节点复算，必须指定 `--run-dir`，不适用于report_blend档案。
- `diagnose_problem4_emergency.py`、`diagnose_problem4_snapshots.py`：20260912旧问题四两月档案的缺口分解和局部复算，脚本绑定该版本；不是新版连续回测。
- `audit_problem4_corrections.py --out <分支结果目录>`：审计v3/v4/v5实际接入、合法节点、压力SOC起点和调减恢复证据，并分别汇总前置/报告期量费。
- `compare_problem4_economics.py --baseline <旧双分支目录> --candidate <新双分支目录>`：核验源数组一致，分解实际费用和购电量变化，分别比较前置、检验与完整运行。
- `compare_a1.py`、`compare_p1.py`：既有重算周期/费用优先级敏感性实验。
- `verify_outputs.py`、`verify_settlement.py`、`verify_timeaxis.py`：旧格式或底层账务时间核验；正式新result使用上级 `verify_strict_result.py`。

不得把历史诊断数据当成新的主模型输出，不读取out/current指针来猜测要使用哪个模型版本。旧结果的运行源码哈希对应清理前快照。
