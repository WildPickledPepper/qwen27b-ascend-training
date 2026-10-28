# 按层扩大 ZeRO-3 参数预取预算

**结论：只把预取预算从 134,217,728 提到最大层的 383,273,184 元素，稳态 109.03 秒/256 条，比保留配置 107.97 秒慢约 1%，不采用。**

## 背景

读源码确认预取元素数受 `min(stage3_max_live_parameters - 可用, stage3_prefetch_bucket_size)` 双重限制，原上限 3,000,000,000 / 134,217,728。既然单层最大 3.83 亿参数，就先把预算提到“一次能覆盖一整层”，看能否减少 AllGather 往返；不改模块划分与梯度 hook。

## 配置

- 双卡物理 2/3，`zero3-sql.json` 的其余项不变；`stage3_prefetch_bucket_size = 383273184`（实测结果 JSON 中的实际值）。
- global256、每卡 micro16、累积 8；BF16 计算、CPU FP32 master/m/v。
- CPU 侧使用已验证的 [32 槽累积版](../sql-gradient-pipeline-20261025/README.md)。

## 做法

`train_layer.py` 用 `list(model.model.layers)`（64 层）分组，取 `sum(p.ds_numel)` 最大值作为预算，并断言 `2 × 最大层 < stage3_max_live_parameters`；`inspect_schedule.py` 只读打印已安装 DeepSpeed 的 `_fetch_sub_module_impl` / `__all_gather_params_` 源码，作为“预取与归约共享流”的依据。

## 结果

| 项目 | 结果 |
|---|---|
| 稳态（同 768 条，前 256 预热后两轮平均折算） | 109.03 秒/256 条 |
| 对照 | 107.97 秒/256 条 |
| 差异 | 慢约 1%，未达采用门槛 |

## 边界

预算变大不等于 AllGather 次数或空等下降：是否受益要看稳态端到端时间与采样后的暴露区间。本轮只改了预算，没有同时解决归约在途约束，因此不能据此否定“按层预取”本身。

## 证据与复现

- 入口 `run_experiment.py` → `train_layer.py`；`build_entry.py` 从 `sql-gradient-pipeline-20261025` 生成本目录代码。
- 原始结果只在本地 `.private` 文件（`rank0/1-result.private.json`、事件文件），本目录**无过滤证据 JSON**；结论引自 [调度研究](../../research/training-scheduling-20261025.md#L27)。
- 后续：归约暴露（ReduceScatter 15.58 秒、AllGather 10.62 秒）确认后，优先级转向 [归约调度](../sql-reduce-schedule-20261025/README.md)，[FFN 分组](../sql-ffn-gather-20261025/README.md) 因此没有上卡。
