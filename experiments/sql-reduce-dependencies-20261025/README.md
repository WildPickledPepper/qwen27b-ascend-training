# 梯度归约暴露：桶、提交、传输与计算下发

分析执行于2026-10-26（Asia/Shanghai），目录/evidence日期沿用2026-10-25采样日期。先复用保留95.7402秒/256条版本的第四轮rank0数据，设备窗口113.481秒。随后事件关联补采另记；本报告前半段没有运行新模型或改默认配置。

## 13.36秒落在哪里

13.361874859秒是归约大算子区间减去全部已记录计算区间的并集，不自动等于可消除等待。按相同区间细分：

| 归约子任务 | 时长 |
|---|---:|
| Reduce_Inline执行区间 | 12.4398秒 |
| 内部Memcpy | 0.6491秒 |
| Notify_Wait | 0.2496秒 |
| Notify_Record | 0.0027秒 |
| 未覆盖的子任务间隙 | 0.0206秒 |

子任务在同一通信流顺序执行，分类之和等于13.361874859秒。Reduce_Inline占约93.1%，实际大数据量子任务走SDMA/HCCS。执行区间含调度与完成响应，不能仅凭比例宣称物理带宽饱和。

## 哪些梯度桶

仅读原checkpoint张量头部，筛选实际训练文本主干，总参数26895998464与记录一致，未加载权重或使用NPU。按每rank HCCL count、原生双卡分片规则及调用次数核对：

| 参数组 | 参数数×8微批 | 归约暴露 |
|---|---:|---:|
| 64层FFN gate/up/down，5120×17408 | 192×8＝1536次 | 7.0801秒 |
| embed_tokens与lm_head，248320×5120 | 2×8＝16次 | 2.3287秒 |
| 48层线性注意力in_proj_qkv，10240×5120 | 48×8＝384次 | 1.8650秒 |
| 其他注意力投影及小参数 | 其余 | 2.0881秒 |

这是训练梯度同步，不与推理权重搬运混为一谈。HCCL count不能直接当完整输入梯度元素数。

## 4504次提交全部匹配

此前按CANN外层`hcom_reduceScatter_`名称过滤导致零匹配，是查询层级选错，不是ms工具无法关联。设备大算子的connectionId实际对应CANN中名为`launch`的记录；外层hcom API与launch起止一致，但ID不同。

现4504次归约唯一匹配4504个launch，不猜ID加减1。最大FFN桶count44564480，从API返回到设备开始平均16.3043毫秒，1528/1536次提前超过1毫秒：

| 已提交后的时间在做什么 | 平均每次 |
|---|---:|
| 前置参数AllGather | 6.6246毫秒 |
| 前置梯度ReduceScatter | 7.7285毫秒 |
| 通信流EVENT_WAIT | 1.8975毫秒 |
| 其他 | 0.0537毫秒 |

不同桶的提前区间会重叠，不能累加成墙钟。AllGather/归约底层任务均在468流，设备区间交集为0；原始HostTask也确认其FFTS_PLUS父任务均为468流。提前提交仍要排在已有工作之后。换通信域能否改善尚未测试，不能由共流许诺收益。

## 为何没有计算重叠

主计算流466的189209个任务全部匹配CANN launch。对13.3619秒逐段查看下一项主流计算：

| 提交状态 | 时长 |
|---|---:|
| 宿主尚未开始提交 | 7.8940秒 |
| 正在提交API内 | 0.0903秒 |
| API已返回，设备尚未执行 | 5.3776秒 |

未开始提交区间的宿主最内层记录包括：梯度分片0.666秒、Event::record0.621秒、ZeRO Linear反向0.547秒、PythonDispatchMode0.459秒、PreBackward钩子0.454秒、AccumulateGrad0.430秒、PostBackward钩子0.393秒，另有未打点及其他范围。这是上下文时间归属，含采样/观测开销，不等于各函数可净省的CPU时间。

已提交部分5.3776秒中，主流直接记录EVENT_WAIT约1.2062秒，另4.1616秒没有对应主流任务记录，不能自行命名为“等待HCCL”或“GPU空闲”。原始数据重导出后，runtime.db里相关主流kernel的HostTask时间戳均在各自launch API内，未发现API返回后才标记任务的情况。原始戳在CPU计数域内比较，未猜作纳秒。

现有数据没有完整保留框架Event对象到生产者的关系，仍缺每个等待的确切生产者。新增双rank采样仅给既有Event.record/wait/synchronize、宿主梯度钩子、桶参数ID加关联标记，保持所选数学/精度/配置；先小模型验证真实设备事件关联，再采整模型。结果完成后追加。

## 方法与决定

使用ascend-profiler-db-explorer原样Compute CTE、官方CANN/通信/TASK字段、只读SQLite。PYTORCH_API、CONNECTION_IDS及原始内部表未在技能文档中提供，才补充真实CREATE TABLE。Git只留轻量汇总/SQL，原始大数据与第三方源码保留私有。

约373MB原始捕获在私有副本上经msprof成功重导出timeline、DB和task_time，未覆盖原采样。[官方交付件说明](https://mindstudio-docs-master.readthedocs.io/zh-cn/latest/msprof/docs/en/user_guide/profile_data_file_references/)要求结合任务与下发连线分析，不将空白自行解释成硬件空闲。

现有定位支持两个具体方向：减少宿主逐参数反向/事件/分片处理对计算下发的阻滞；核对共享队列中的大梯度与参数预取安排。收益须另测，不能把采样窗口的7.89/13.36秒直接从正常95.74秒扣除。默认未改，本阶段无提速声明。

证据：[传输分解](../../evidence/2026-10-25/sql-reduce-dependencies.json)、[队列](../../evidence/2026-10-25/sql-reduce-queue.json)、[计算提交](../../evidence/2026-10-25/sql-reduce-compute-readiness.json)、[参数核对](../../evidence/2026-10-25/sql-reduce-native.json)。
