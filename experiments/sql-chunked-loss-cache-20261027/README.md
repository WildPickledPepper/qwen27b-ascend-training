# 用chunked loss释放的显存扩展梯度驻留（进行中）

2026-10-27用户授权。在实时空闲物理3/4号910B2上，最终512-token分块＋连续化loss保持，rank0/3号缓存4GiB不变，rank1/4号由4→8GiB。按原参数贪心选择，实际恰好8GiB、77个完整参数梯度（旧29个）。不改训练batch、更新边界、48层重计算、BF16累加顺序或CPU FP32 master/m/v。

新增4GiB在16micro期间留NPU逐次累加、末尾只卸载一次，理论上每次更新少4×(16−1)=60GiB D2H；GPU预算增加不减少CPU主权重/m/v容量。以49.835765GiB实测峰值加持久4GiB估算约53.84GiB，另保留allocator余量，不把估算当放得下证明。

复用刚完成的4/4GiB contiguous-loss三步对照，不重跑基线。tiny三步完整grad/master/m/v全部逐位相同后才启动27B原768条三步；完整实验仍在进行。原81.22秒入口及上轮4/4源码不改，独立目录运行，结束清理自己进程。

`train_cache.py`只新增每rank预算参数；`resident_gradients.py`在新副本中将授权预算上限改为8GiB，其余保序逻辑不变。`run_sequence.py`先检查3/4空闲、tiny及逐位门槛，再完整3步；环境提供既有MODEL_DIR、DATA_DIR、CPU_ADAM_EXTENSION、VEOMNI_SOURCE、TINY_REFERENCE（上轮tiny-contiguous输出）。不涉及共享库或模型文件修改。

`collect_results.py`核对相同数据/FP32状态、逐步D2H差额、驻留调用、范数和验证，输出过滤聚合结果。CPU累加/等待等存在重叠，不能相加凑整步。
