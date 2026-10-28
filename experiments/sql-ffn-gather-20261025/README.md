# FFN 作为 leaf 模块分组 + 匹配预取（未上卡）

**结论：没有实测结果。** 本目录只完成代码准备，控制器在等待前序任务阶段就被主动取消，`job.json` 记为 `cancelled_before_gpu_use`；不得把任何数字或加速归到本方案。

## 方案

用原生 `set_z3_leaf_modules(model, [type(model.model.layers[0].mlp)])` 只把 64 个 FFN 注册为 leaf，让 FFN 成为参数管理与汇聚/释放单位（而不是每层一次），再把 `stage3_prefetch_bucket_size` 设成实际最大 FFN 的参数量。目的是把零散的参数 AllGather 收拢成更完整的一批，而不是全模型一次汇聚。

## 为什么在启动前取消

`cancel_waiting.py` 里的取消理由是原文照录的决策依据：

> Prioritize measured exposed ReduceScatter; FFN leaf grouping may delay gradient readiness.

也就是说：当时已经实测到 AllGather 暴露 10.62 秒、ReduceScatter 暴露 15.58 秒，归约在途是更确定的瓶颈；而 leaf 策略会改变 FFN 梯度回调时机，可能牺牲一部分归约重叠，收益不确定却要占卡时间。因此本方案让位给 [归约调度](../sql-reduce-schedule-20261025/README.md)。

## 目录里有什么

| 文件 | 作用 |
|---|---|
| `build_entry.py` | 由 `sql-layer-gather-20261025` 派生本目录，加 `--ffn-leaf` 分支，并在最后恢复上一轮入口源码 |
| `train_ffn.py` | 含 `set_z3_leaf_modules` 的 FFN 分组版训练入口（`expected=64` 断言） |
| `cancel_waiting.py` | 只取消本控制器、要求尚未启动任何子进程的守卫脚本 |
| `run_experiment.py` | 串行控制器：等待上一轮 job 结束后 tiny → 逐位比对 → 整模型 4 更新 |
| `launch.private.sh` | 本机启动脚本（含私有路径，未入库） |

## 状态与风险

- 状态：**仅代码准备 / 未上卡**。任何“FFN 分组更快/更慢”的说法都缺依据。
- 复现前提：`run_experiment.py` 依赖上一轮 `job.json` 与 `tiny-native` 软链；在没有前序任务的环境里需要先跑 `sql-gradient-cache-20261025` 的 tiny 参考，否则会在等待阶段超时。
- 若要重做：需要同时测“汇聚更完整”的收益与“梯度回调推迟”的损失，并检查 leaf 下梯度 hook 时机是否仍与原归约重叠。
