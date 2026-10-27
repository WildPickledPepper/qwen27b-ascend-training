# Chunked loss在物理3/4号910B2的实测（进行中）

用户2026-10-27授权在3/4号卡实测。两卡初始化、BF16矩阵乘、HCCL和P2P通过；输出层小形状及真实V248320/H5120、短token前后向通过；tiny流水线三步完整梯度/master/m/v对比通过。完整27B配对实验进行中，没有显存收益或速度完成声明。

固定原SQL768条、顺序/padding/global256/micro16/16micro、48层非重入重计算、4GiB BF16梯度驻留、CPU FP32主权重与Adam m/v。独立`train_chunked.py`由原81.22秒入口派生；只改变训练loss路径及卡组/对应CPU-NUMA绑定。3号卡NUMA4/CPU96–119，4号卡NUMA0/CPU0–23。验证仍用完整普通loss与logits，方便同口径统计；每次更新边界和全局有效回答token加权保持。

普通loss和512-token分块各3次更新，第一步lr0预热，后两步计时；记录两rank每步allocated/reserved峰值、范数、卸载字节、验证与数据哈希。必须在同卡组对照，不把历史2/3卡81.22秒直接作速度参照。新完整模型保存不在范围内，原模型/共享环境不改，完成释放自身任务。

## 预检与失败

初版tiny使用原逐元素allclose（rtol2%、atol1e-7），lm_head.weight因BF16分块求和顺序改变失败，最大绝对差1.220703125e-4；rank0其他梯度逐位一致。原始失败保留在`train_initial.py`与服务器日志，未绕过后直接跑27B。

修正仅涉及tiny验证口径：lm_head保留逐元素比较结果，增加并使用与输出层预检相同的整体relative L2<2%及有限值检查；其他参数维持原逐元素门槛。没有修改BF16累加数学。tiny三步所有张量对照最大整体梯度relative L2约0.0026151、master约3.452e-7；不等于无损或长期收敛认证。

真实输出层尺寸短输入loss绝对差1.91e-6，输入/权重梯度relative L2约3.47e-6/6.07e-4。详见[预检证据](../../evidence/2026-10-27/sql-chunked-loss-device-preflight.json)。

实现来自已建模的`chunked_loss.py`：前向分块线性+FP32 CE，反向逐块重算、BF16部分权重梯度累加；不保留所有块logits计算图，但仍有完整lm_head梯度和临时部分梯度。不使用新精度、压缩或新优化器。只验证这套普通PyTorch实现，不冒称NPU融合loss内核。
