# Chunked loss在物理3/4号910B2的实测（进行中）

用户2026-10-27授权在3/4号卡实测。两卡初始化、BF16矩阵乘、HCCL和P2P通过；输出层小形状及真实V248320/H5120、短token前后向通过；tiny流水线三步完整梯度/master/m/v对比通过。完整27B配对实验进行中，没有显存收益或速度完成声明。

固定原SQL768条、顺序/padding/global256/micro16/16micro、48层非重入重计算、4GiB BF16梯度驻留、CPU FP32主权重与Adam m/v。独立`train_chunked.py`由原81.22秒入口派生；只改变训练loss路径及卡组/对应CPU-NUMA绑定。3号卡NUMA4/CPU96–119，4号卡NUMA0/CPU0–23。验证仍用完整普通loss与logits，方便同口径统计；每次更新边界和全局有效回答token加权保持。

普通loss和512-token分块各3次更新，第一步lr0预热，后两步计时；记录两rank每步allocated/reserved峰值、范数、卸载字节、验证与数据哈希。必须在同卡组对照，不把历史2/3卡81.22秒直接作速度参照。新完整模型保存不在范围内，原模型/共享环境不改，完成释放自身任务。

## 预检与失败

初版tiny使用原逐元素allclose（rtol2%、atol1e-7），lm_head.weight因BF16分块求和顺序改变失败，最大绝对差1.220703125e-4；rank0其他梯度逐位一致。原始失败保留在`train_initial.py`与服务器日志，未绕过后直接跑27B。

修正仅涉及tiny验证口径：lm_head保留逐元素比较结果，增加并使用与输出层预检相同的整体relative L2<2%及有限值检查；其他参数维持原逐元素门槛。没有修改BF16累加数学。tiny三步所有张量对照最大整体梯度relative L2约0.0026151、master约3.452e-7；不等于无损或长期收敛认证。

真实输出层尺寸短输入loss绝对差1.91e-6，输入/权重梯度relative L2约3.47e-6/6.07e-4。详见[预检证据](../../evidence/2026-10-27/sql-chunked-loss-device-preflight.json)。

实现来自已建模的`chunked_loss.py`：前向分块线性+FP32 CE，反向逐块重算、BF16部分权重梯度累加；不保留所有块logits计算图，但仍有完整lm_head梯度和临时部分梯度。不使用新精度、压缩或新优化器。只验证这套普通PyTorch实现，不冒称NPU融合loss内核。

## 新目录复现协议

将本目录代码与固定模型/数据/CPUAdam/VeOmni环境放入新实验目录，指定`MODEL_DIR`、`DATA_DIR`、`CPU_ADAM_EXTENSION`、`VEOMNI_SOURCE`，运行`python run_sequence.py`。它先检查3/4卡空闲，依次初始化/输出层/两个tiny/完整状态比较，然后才完整27B对照与512分块；每阶段均有超时和失败即停。`run_initial_sequence.py`/`train_initial.py`保留首次失败代码，不用于新运行；`resume_sequence.py`是本轮保留初始失败后续跑入口。运行时source-manifest记录源码SHA，日志和私有快照不提交。

本轮既有对照从头运行，无隐含继续训练；tiny实验用随机小模型，完整模型从原权重各自开始。没有保存27B更新后checkpoint。

## 第一轮整模型结果：未采用，布局修正版在测

三步普通loss与初版512分块均完成，同一3/4卡、同768条数据/顺序/padding，参数、FP32状态字节、梯度D2H量和驻留累加次数一致。普通loss后两步约77.99秒，初版分块约85.51秒，慢9.65%；rank1 allocated峰值54.054480→54.572093GiB，增加0.517613GiB。reserved有所下降不能等同活跃内存下降。验证loss变化约-0.002013，只是8条短验证。初版不采用，不将原6.35GiB预测冒充实测。

随后在完整输出头形状B16/S256/H5120/V248320单独同步观测：自动求导产生的BF16权重梯度stride[1,248320]非连续。第一次`dw.add_(gw)`前当前allocated约7.187567GiB、此前峰值7.690498GiB；加法后当前allocated不变，瞬时峰值14.292061GiB，额外临时空间7.104494GiB，恰相当三份梯度大小。仅凭这一差额不声称内部一定有三个特定副本，但证据将峰值直接定位到非连续梯度加法调用；meta对原地add没有记录到这些后端临时分配。

`chunked_loss_contiguous.py`只在取得部分梯度后添加`gw.contiguous()`，保持BF16和分块顺序，显式复制后再累加。同尺寸单独测试中梯度stride[5120,1]连续，加法不再增加峰值，整体峰值14.292061→9.555733GiB。明确连续化也需要一份完整临时梯度，不能当成零成本。正在重新通过输出层、tiny及整模型三步验证；不修改原入口、缓存预算或精度。

[首轮结果](../../evidence/2026-10-27/sql-chunked-loss-device-first-results.json) · [非连续加法观测](../../evidence/2026-10-27/sql-chunked-loss-layout-diagnostic.json) · [连续化观测](../../evidence/2026-10-27/sql-chunked-loss-layout-contiguous-diagnostic.json)
