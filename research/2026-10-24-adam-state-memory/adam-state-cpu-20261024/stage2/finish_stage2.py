"""Audit completed outputs and build a transfer-ready stage report; no new runs."""
import csv,hashlib,heapq,json,math,struct,sys,tracemalloc
from pathlib import Path
import run_stage2 as s
e=s.e;c=s.c;root=Path(__file__).resolve().parent
read=lambda name:list(csv.DictReader((root/name).open(encoding='utf-8')))
replay=read('replay_summary.csv');toy=read('toy_summary.csv');protect=read('protection_summary.csv')
assert len(replay)==162 and len(toy)==30 and len(protect)==8
for x in toy:
    assert math.isfinite(float(x['final_loss']))
    if x['kind']=='fp32':assert float(x['final_weight_RMS'])==0 and float(x['final_loss_ratio'])==1
assert all(float(x['projected_peak_GiB_including_workspaces'])<=30 for x in protect)
# Verify real packed fields preserve the unquantized half; no hidden FP32 state.
g=next(e.gradients('random',256,1,20261024));m,v=e.moments([0.]*256,[0.]*256,g)
for k in ('solo42','zip_sr','ze_eden'):
    for mode in ('m','v','both'):
        b=s.store(m,v,k,mode,17,1);mm,vv,_=s.restore(b,k,mode)
        if mode=='m':assert e.pack(v)==e.pack(vv)
        if mode=='v':assert e.pack(m)==e.pack(mm)
        if mode=='both':assert b==c.encode(m,v,k,17,1)
mem=[]
for k in ('zip_sr','solo42'):
    n=16384;initial=s.store([0.]*n,[0.]*n,k,'both',17,0);g=next(e.gradients('random',n,1,20261024))
    tracemalloc.start();m,v,r=s.restore(initial,k,'both');m,v=e.moments(m,v,g);d,w=e.update(m,v,[.1]*n,1);blob=s.store(m,v,k,'both',r,1);mm,vv,_=s.restore(blob,k,'both')
    def scores():
        for i in range(0,n,128):
            dd,_=e.update(mm[i:i+128],vv[i:i+128],[0.]*128,1)
            yield (e.err(dd,d[i:i+128])[1],i//128)
    entries=heapq.nlargest(2,scores());_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    entry_bytes=sum(sys.getsizeof(z)+sum(sys.getsizeof(a) for a in z) for z in entries)+sys.getsizeof(entries)
    mem.append({'kind':k,'n':n,'actual_compact_blob_bytes':len(blob),'tracemalloc_scratch_peak_bytes':peak,'measurement':'one encode/decode/update/scoring window; initial compressed state and gradient allocated before tracing; no reference arrays','two_selected_heap_entries_total_python_bytes':entry_bytes,'design_topk_allowance_bytes_per_entry':128,'design_fixed_allowance_bytes':8*2**20,'fits_fixed_allowance_this_CPU_window':peak<8*2**20})
assert all(x['fits_fixed_allowance_this_CPU_window'] for x in mem)
audit={'all_assertions_passed':True,'replay_configurations':len(replay),'toy_configurations':len(toy),'protection_configurations':len(protect),'current_stage_vector_steps':162*256+30*1024+8*48,'superseded_format_only_vector_steps':8*48,'memory':mem,'stopped_after_current_stage':True,'no_new_training_or_remote_access':True}
(root/'audit_stage2.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
lines=['# 第二阶段：误差消融、闭环凸优化与预算保护','',
'**当前结果：m 量化是重要误差来源；低位 v 处理改善不能单独解决问题。** 本地 DE-m + ZIP-SR/ZE-EDEN-v 的闭环 toy 仍明显弱于相同超参 FP32。受30GiB预算限制的异常块保护仅部分改善固定梯度重放，尚无27B可用性证据。按最新指令完成此阶段后暂停，供迁移到云端工作区。','',
'本阶段没有重新访问原仓库，也没有克隆或连接远程计算环境。首阶段 ZIP 已成功上传 Library，ID：`[archive listed in ARTIFACT_INDEX.json]`。Windows扩展属性写入失败，身份另存JSON；不影响Library上传成功。','',
'## 统计单位与配置','',
'当前版本包含41,472个固定梯度向量步骤、30,720个闭环toy优化步骤、384个保护对照步骤。每步仅处理256或16384个CPU标量，**不是27B训练步**。另保留384个已废弃的格式预算保护步骤，见`format_only_*`，不能混入当前预算结论。','',
'- 重放：n=256、256步、6类梯度；3算法×m-only/v-only/both×3舍入种子17/29/43。固定相同梯度与超参。','- 闭环：n=256、1024步、3数据/初始化种子101/202/303；目标是正定对角二次函数 L=Σ aᵢ(wᵢ-targetᵢ)²/(2N)，a跨1e-2至1e2。每种配置从同一初始化出发，梯度由其自身当前权重计算；没有数据划分或神经网络任务。','- 所有运行保留参考beta=(.9,.999)、lr=1e-4、eps=1e-8、wd=.01以及FP32主权重和原运算顺序；toy衡量的是固定1024步后的目标值，不是已收敛最优值。','- 每步relative L2和cosine针对完整更新向量；p50/p95/p99针对当步坐标绝对误差，汇总再取跨步骤最大值。cosine汇总取跨步骤最小值。v零化累计单位为“坐标×步骤”，不是不同参数数目。权重最终RMS和最大绝对偏差单列。','- m-only/v-only确实只序列化被量化一侧的码，另一侧按FP32保存；这两项用于消融，不能当作30GiB达标方案。','',
'## 随机梯度重放：种子17的可核对结果','',
'| 算法 | 量化侧 | 最大更新relative L2 | 最小cosine | 最终权重RMS偏差 |','|---|---|---:|---:|---:|']
for x in replay:
    if x['case']=='random' and x['rounding_seed']=='17':lines.append(f"| {x['kind']} | {x['mode']} | {float(x['update_rel_L2_max_over_steps']):.6g} | {float(x['update_cosine_min_over_steps']):.6g} | {float(x['final_weight_RMS']):.6g} |")
lines+=['','三种方法的m-only结果相同，符合它们共用同一个DE m码本的设计。ZIP-SR在稀疏尖峰、种子17时累计653次正v→零事件；ZE-EDEN和SOLO式该序列无此事件，但仍有显著误差，说明“没有零化”不等于可靠更新。','',
'## 闭环toy：三个固定种子的范围','',
'| 算法 | 量化侧 | 最终目标值 / 配对FP32目标值范围 |','|---|---|---:|']
for k in ('solo42','zip_sr','ze_eden'):
    for mode in ('m','v','both'):
        vals=[float(x['final_loss_ratio']) for x in toy if x['kind']==k and x['mode']==mode]
        lines.append(f'| {k} | {mode} | {min(vals):.4f}–{max(vals):.4f} |')
lines+=['','FP32最终目标值范围为'+f"{min(float(x['final_loss']) for x in toy if x['kind']=='fp32'):.8g}–{max(float(x['final_loss']) for x in toy if x['kind']=='fp32'):.8g}"+'。只量化m已明显阻碍本toy优化，下一步应优先改变m编码/舍入策略。ZIP-SR、ZE-EDEN这里只复现v规则，m使用DE；正式论文采用NF4和局部随机舍入，因此不能用此结果否定论文完整方法。SOLO式保持beta1=.9且使用自定零分位数边界策略，也不是论文完整配置复现。','',
'## 异常块保护：把切换峰值也计入30GiB','',
'先前`format_only_*`只算稳态回退加1MiB，漏掉切换时旧/新回退与选择结构，已明确废弃。修正版n=16384、窗口128、48步、固定种子17；持久存储保留完整压缩码，再用异常块FP32覆盖，两套回退缓冲同时计费。每条异常记录实际编码uint32块索引4 bytes+两组128元素FP32共1024 bytes；每套另有8-byte计数。全局24-byte头部含步数/形状/PRNG，码本64或128 bytes。','',
'目标参数数N=26,895,998,464，总预算公式：基础压缩payload + ceil(实测保护比例×ceil(N/128))×[2×(1024+4)+128] + 全局头/双计数/共享码本 + 8MiB固定工作区。128 bytes/选中块覆盖有界top-k选择结构；不保留全模型每块分数表。8MiB是本CPU窗口审计可覆盖的设计保留量，不是NPU峰值测量。生产实现仍须原位分块编码，不能全模型双份解码。','',
'| 梯度 | 算法 | 保护比例 | 总设计峰值GiB | 最大更新relative L2 | 最终权重RMS |','|---|---|---:|---:|---:|---:|']
for x in protect:lines.append(f"| {x['case']} | {x['kind']} | {100*float(x['max_protected_fraction']):.4f}% | {float(x['projected_peak_GiB_including_workspaces']):.6f} | {float(x['update_rel_L2_max_over_steps']):.6g} | {float(x['final_weight_RMS']):.6g} |")
lines+=['','CPU单窗口实际tracemalloc峰值、序列化字节数及审计结果见[audit_stage2.json](audit_stage2.json)。全模型保护选择和索引容量是按该格式比例计算；这不是27B实测，也未含模型权重/梯度/激活。8MiB固定工作区不证明Ascend分配器或融合内核能满足预算。','',
'## 复现与文件','',
'从研究根目录运行：','```powershell','python stage2/run_stage2.py --part replay','python stage2/run_stage2.py --part toy','python stage2/run_stage2.py --part protect','python stage2/finish_stage2.py','```','',
'当前`run_stage2.py`只修正了保护预算和选择结构，replay/toy函数不变；它们运行时的完整脚本快照是`run_stage2_format_only_snapshot.py`，对应run_replay/run_toy记录中的哈希。不能用快照重新运行保护并当作最新结果。','',
'结果：`replay_summary.csv`/`replay_per_step.csv`；`toy_summary.csv`/`toy_per_step.csv`/`toy_inputs_sha256.json`；`protection_summary.csv`/`protection_per_step.csv`；三个run JSON；`audit_stage2.json`。逐步表保留所有分位、最大误差、cosine、v零化和权重偏差。','',
'## 迁移后下一优先项（尚未启动）','',
'先替换m的DE确定性码本为NF4及有依据的随机舍入，保持v规则不变重跑这组消融与闭环toy；再验证异常块阈值、长期稳定性及真实已有状态样本。不要先扩到27B。当前所有计算均已完成并停止；无原仓库重试，无NPU或远程访问。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
