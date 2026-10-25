# CPU卸载更新与训练算子融合：2026-10-25

用户要求沿CPU卸载更新与训练算子融合继续研究。沿用物理2/3号910B2、BF16计算、CPU FP32 master/m/v、原AdamW及独立训练/验证划分；原checkpoint只读，无新Git提交/推送授权。不修改共享安装或其他服务。

## 已有证据与研究顺序

[上一轮](../dual-throughput-20261025/README.md)同全局batch2的双卡优化后30.1349秒/更新；同全局batch32的最好结果34.8777秒/更新。rank0采样CPUAdam本体9.729秒、144031个计算任务累计6.519秒。阶段/跨流事件会重叠，不用事件累计相加凑墙钟。

本轮分别验证CPU冗余遍历、CPU→NPU流水、社区训练GDN融合，再接整模型，避免同时改变多项后无法判断收益。CPU-only改动要求4步最终验证loss与旧组差小于1e-5、正确token仍263；融合允许BF16中间计算的舍入差异，短跑筛查门限为loss差小于0.02、正确token变化不超过8。这只是早期筛查，不代表长期收敛质量通过。

## CPU更新

`unscale_and_clip_grads`即使loss_scale=1且clip_grad=0，仍遍历所有FP32梯度做乘以1。进程内适配只在这两个标量条件成立时跳过，其余情况回到原函数；原溢出检查、范数汇总、梯度归约和CPUAdam保持不变。

位级小测包含16,777,224个随机/边界坐标，乘以1前后差0；相同CPUAdam连续3次更新后master/m/v位差全部0。测试不是整模型梯度逐位证明。

原DeepSpeed创建的CPU FP32 master未锁页；直接把non_blocking设True不能据此宣称有异步流水。本轮另设两份锁页BF16缓冲，CPU先转换，再异步复制到原NPU BF16分片，和下一组CPUAdam重叠。复用槽位前等待对应Event，`_post_step`进入持久参数收集前统一等待全部复制完成。保留FP32 master与m/v，不修改Adam数学实现。

覆盖BF16舍入边界的326,400个有限FP32坐标，CPU预转换与原直接复制到NPU BF16位差0。8组×3轮、两缓冲重复复用，BF16位差0。复制submit、CPU转换、复用等待及末尾等待分项计时；submit时间不是DMA耗时。

首次正式加载在源码保护检查处退出，尚未更新模型：DeepSpeed的NVTX装饰器没有保留`__wrapped__`，inspect读到外壳。修复为只沿已知NVTX闭包找到真实函数，再验证源码特征；原失败代码和监督记录保留，不当性能结果。

## 社区训练融合

来源：[VeOmni训练指南](https://veomni.readthedocs.io/en/latest/examples/qwen3_5.html)，仓库固定提交`eba362dd4ca43b6cba53dc69c1d378cf3eda57b6`。只在独立目录读取原始训练kernel，没有安装整个VeOmni或改动其源码。

复用`veomni/ops/kernels/gated_delta_rule/chunk_gated_delta_rule/standard/triton.py`的原始forward/backward及vendor Triton-Ascend kernel；适配器提供SavedState值类型和PyTorch autograd连接，避免导入无关训练器依赖。上游文件保留其原始版权/许可证。当前环境torch/torch_npu2.10、triton-ascend3.2.2、triton3.5.0、CANN9.1，实际编译目标Ascend910B2。

最初3组小张量（batch/length/heads=1/32/4、1/128/4、2/64/4，head_dim128）均完成前后向：输出relative L2约0.00360–0.00371，各输入梯度约0.00232–0.00637，均有限。原HF分解会将GDN中间计算转FP32，社区融合使用BF16矩阵计算，不能称逐位等价，也不把FP32 master/m/v保留误写成全模型计算逐位相同。

实际模型48个value head，适配器仅支持本次长度≤128、无cache/无initial_state/无varlen的训练。GDN内部在未来位置补零到128，再裁回有效输出；因因果依赖不改变有效位置的数学计算，只为固定Triton编译形状，主干其他模块不补齐到128。上板再验证实际形状输出/梯度及预热速度，编译在集合通信启动前单独完成，不把JIT时间当稳态速度。

`community_gdn_unpadded.py`保存最初小测适配，`community_gdn.py`为固定形状适配；`warm_gdn.py`测试实际48头及batch16形状。前后向小测只是接入门槛，最终以整模型验证和速度为准。

## 代码与结果

`build_entry.py`从上一轮冻结入口派生`train_fusion.py`；`cpu_update.py`为进程内CPU优化与分项计时；`run_research_v2.py`顺序执行正式研究；`collect_results.py`只导出聚合指标/哈希/源码版本，不导出消息全文或权重。全部正式组保留FP32 master/m/v与训练验证分离，完整结果及结束状态见下方。

CPU冗余遍历组4步后验证loss1.504419923、263/433，与原结果相同；后3步30.0193秒，对原30.1349秒差约0.4%，不足以据此宣称显著整步加速。每步44次identity scaling全部跳过，累计约0.4毫秒。分项中CPUAdam约10.4–10.7秒、同步权重回传约3秒；与旧采样存在运行波动，不能把旧`mul_`累计耗时直接当成已实现的整步收益。

异步路径记录每rank每步26,895,998,464字节的BF16提交量；旧DB没有`MEMCPY_INFO`，尚未核对原复制内部实际H2D字节，因此不能仅凭FP32源张量就宣称PCIe数据量一定减半。

异步回传4步完整通过，后3步28.5143秒，对本轮同步组30.0193秒吞吐约1.053倍。rank0预热CPUAdam约10.67–10.84秒，CPU BF16转换约1.55秒，复制提交约0.022秒，两缓冲复用等待约0.002秒、末尾等待约0.067秒；不把提交时间写成DMA传输时间。最终loss1.504420042、263/433，较原组差约−1.19e−7，NPU分配峰值仍每卡约40.67 GiB。

## 整模型结果

均从同一原checkpoint开始，排除首个更新，取两rank较慢的一方。batch2每组相同8条训练/4更新；batch32每组相同32条训练重复3次更新。批量间不作相同训练配方的速度比。

| 全局batch | 配置 | 稳态秒/更新 | 有效输入token/s | 最大分配峰值GiB/card | 验证loss | 正确token |
|---:|---|---:|---:|---:|---:|---:|
| 2 | 上一轮最好配置 | 30.1349 | 2.400 | 40.67 | 1.504420 | 263/433 |
| 2 | 本轮仅去掉identity scaling | 30.0193 | 2.410 | 40.67 | 1.504420 | 263/433 |
| 2 | 加锁页BF16异步回传 | 28.5143 | 2.537 | 40.67 | 1.504420 | 263/433 |
| 2 | 再加社区训练GDN融合 | **25.4962** | **2.837** | **39.13** | **1.499013** | **266/433** |
| 32 | 上一轮最好配置 | 34.8777 | 68.15 | 50.50 | 1.446561 | 271/433 |
| 32 | 异步回传+GDN融合，同48层重算 | **27.5756** | **86.20** | **50.50** | **1.447144** | **271/433** |

batch2相对上一轮吞吐约1.182倍、耗时−15.39%；batch32约1.265倍、耗时−20.94%。融合后的batch2初始验证loss2.056355、228/433，与原2.048570、229/433已有差异，说明计算路径确有数值变化。短跑后loss下降/正确token略多不代表长期质量优于原实现。

大批量全关激活重算仍OOM，保留失败并使用与旧组相同的48/64层重算；没有把不同重算层数的收益混入最终batch32比较。

实际48头单层稳态前后向（同步后计时，每种3次）batch1/length31：0.0870→0.0156秒；batch1/length128：0.0877→0.0144秒；batch16/length125：0.0898→0.0288秒。前两者约5.6–6.1倍，后者约3.1倍。首次执行0.23秒等不作稳态比较；此前口头“十几倍”已经更正。单层只占整步一部分，不能将它的倍数当整模型倍数。

实际大向量最初使用CPU FP32累计计算cosine，出现大于1的无效数值；原记录保留并标警告。训练全部退出后重跑相同形状，指标用CPU FP64重算：输出relative L2约0.003645–0.003703，最差梯度约0.005869，最低cosine约0.9999828，均有限并通过原门限。最终精度结论只使用FP64复核及整模型验证，不使用原不可靠的cosine值。

## 同口径Profiler对比

依照`mindstudio_profiler_data_check`先检查rank0 PyTorch框架采样的Stop元数据/DB，之后使用`ascend-profiler-db-explorer`的CTE，只读SQLite执行（当前没有配置msprof_mcp）。新入口采用一个active的RECORD_AND_SAVE调度，未沿用旧手动RECORD停止方式。记录训练窗口28.570秒、设备TASK窗口28.593秒，差0.023秒；采样组不参与无采样速度排名。

比较对象是上一轮与本轮相同batch32、相同3更新/96次样本曝光、同48层重算的最后一次更新，均rank0 Level1、CPU+NPU，无shape/stack/memory采集。

| 指标 | 前一轮 | 本轮 | 解读 |
|---|---:|---:|---|
| 计算任务数 | 144,031 | 33,199 | 减少76.95%，小任务明显减少 |
| MatMulV3调用数 | 1,443 | 1,443 | 主干矩阵乘工作未减少 |
| MatMulV3累计秒 | 1.767 | 1.801 | 波动接近，没有矩阵乘本体加速证据 |
| 计算算子累计秒 | 6.519 | 5.481 | 不是墙钟时间拆分 |
| 计算区间去重覆盖秒 | 6.268 | 4.728 | 仅计算任务区间，不等于整卡利用率 |
| CPUAdam本体累计秒 | 9.729 | 10.702 | 数学实现未改，仍是主要瓶颈 |
| AllGather次数 | 857 | 857 | 通信次数保持不变 |
| ReduceScatter次数 | 563 | 563 | 通信次数保持不变 |
| AllGather/ReduceScatter累计秒 | 3.349 / 3.145 | 3.194 / 3.129 | 包含等待，不是纯链路传输时间 |
| host `aten::copy_`累计秒 | 6.754 | 2.263 | 涉及多种复制，不能全部归因为权重回传 |

新DB出现`prepare_wy_repr_bwd_kernel`、`chunk_fwd_kernel_o`等社区训练kernel，host `GDN`96次、`GDNBackward`48次，证明前后向确实接入，而不是只改了配置标签。Mul调用18,834→4,098，Slice17,874→786，符合移除原GDN分解小算子链的目标。

当前最大单项是CPUAdam约10.7秒；`AccumulateGrad`宿主范围约5.5秒，包含ZeRO梯度处理、卸载和同步，不能直接当成纯D2H传输时间。后续应拆分梯度处理并研究CPU更新与反向重叠。当前每组更新仍在反向结束之后执行，只将权重回传与下一组CPUAdam重叠。

## 后续社区方向（未部署）

[ReFlow贡献提案](https://github.com/deepspeedai/DeepSpeed/issues/8697)与当前瓶颈直接相关：保留BF16梯度跨设备传输，CPU消费时在寄存器转FP32，并把CPU更新分为先生成计算权重、通过全局检查后再持久化FP32状态两阶段，以便与反向重叠，同时处理裁剪/溢出。作者的性能与位级一致性结论属于其报告；其列举平台为x86 CPU与NVIDIA GPU，尚不能当作我们Kunpeng ARM+Ascend的可直接启用方案。本轮未部署ReFlow或自写新的CPUAdam数学内核。

[DeepSpeed Twin-Flow](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-offloadpp/README.md)提供CPU/GPU混合状态放置，也是候选路线；它需要显存预算及NPU端FP32 Adam的数值/性能验证。本轮没有开启混合状态或把其他卡加入训练。

## 交付与结束

共5组成功完整结果（含1组独立Profiler），保留首次源码保护检查失败及大批量全关重算OOM。全部正式更新的FP32 m/v总量始终215,167,987,712字节，FP32主权重断言通过；数据哈希/更新次数与上一轮逐一核对，训练验证分离。没有保存新完整checkpoint，未改原权重或共享安装。

全部任务及最终FP64复核正常退出后，检查物理2/3卡无活动进程，基础HBM3416/3415 MiB，其他服务未动。仅更新工作区，没有自动Git提交/推送，也不自动继续训练。

- [逐组过滤证据与小测](../../evidence/2026-10-25/training-fusion-results.json)
- [同数据/状态/数值审计](../../evidence/2026-10-25/training-fusion-verification.json)
- [Profiler SQL与结果](../../evidence/2026-10-25/training-fusion-profile.json)
- [采样对比](../../evidence/2026-10-25/training-fusion-profile-comparison.json)
- [资源释放](../../evidence/2026-10-25/training-fusion-resource-release.json)
- `launch_best.sh`保存batch32复现配置；需要同一CPU/NUMA布局、空闲2/3卡以及指定提交的社区源码，冷缓存先单独预编译。

这轮证明短序列性能和短跑数值检查通过，未证明长期训练质量或达到硬件上限。
