# CPU BF16 梯度累积（双有界接收缓冲）

**结论：逐位与验证都通过，但稳态 119.42 秒/256 条仍慢于原配置的 107.97 秒，不采用。** 它测出的“主线程等接收槽 + CPU 相加都在关键路径”成了下一轮 32 槽方案的直接依据。

## 背景

上一轮 [12 GiB 梯度常驻 NPU](../sql-gradient-cache-20261025/README.md) 因物理页申请/释放开销反慢到 126.66 秒。本轮改从 CPU 侧下手：保留逐 micro 的 BF16 相加顺序，把“8 轮逐段回传 + 相加”压成接收队列 + 单 worker。

## 配置

- 双卡物理 2/3，`zero3-sql.json`；global256、每卡 micro16、累积 8。
- BF16 计算与 BF16 梯度累积；CPU 侧 FP32 master/m/v 与原生 Adam 不变。
- 每 rank 24 个 CPU 线程，NUMA 就近绑核，`--cpu-mode pipeline --cpu-gradient-accum`。
- 数据沿用原 768 条与 padding 设置，第 4 轮仅采样并排除测速。

## 做法

`cpu_accumulation.py` 用两个有界锁页缓冲交替接收每个 micro 的 BF16 梯度分片，单个 CPU worker 按 FIFO 相加；边界仍走原生 NPU 梯度范数与 CPU FP32 卸载，不增加 NPU 常驻。

## 结果

| 项目 | 结果 |
|---|---|
| tiny 三更新完整梯度/master/m/v | 与原生逐位一致 |
| 整模型 768 条后验证 | 与原生完全一致 |
| 第 2/3 次更新 | 117.68 / 121.16 秒（平均 119.42 秒/256 条） |
| 主线程等接收槽 | 每轮约 17.5 秒 |
| 边界交接 | 另约 9.8 秒 |
| CPU 相加累计 | 约 24 秒 |

## 为什么不采用

队列只有 2 个槽，NPU 侧一旦有多个分片就绪就只能等；等待与 CPU 相加都落在同一条关键路径上。结论与后续拆解见 [固定 256 条调度研究](../../research/training-scheduling-20261025.md)。

## 证据与复现

- 入口：`run_experiment.py`（控制器，串行占用 2/3，先 tiny 与 `compare_probes.py` 逐位比对，再整模型）；`build_entry.py` 从 `sql-gradient-cache-20261025` 生成 `train_cpu.py`。
- 原始 `rank0/1-result.private.json` 与事件文件是本地产物，未入库；本目录**没有**过滤后的证据 JSON，数字引自调度研究文档与授权归档。
- 下一轮：[32 槽有界接收](../sql-gradient-pipeline-20261025/README.md)。
