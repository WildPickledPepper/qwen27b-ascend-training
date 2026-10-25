"""Build a protocol-aware overview from saved evidence; no hardware activity."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def read(name):return json.loads((root/'evidence'/name).read_text())
precision=read('2026-10-24/precision-compare-results.json')
pipeline=read('2026-10-24/fp32-cpu-master-pipeline-results.json')
mixed=read('2026-10-24/bf16-master-training-results.json')
dual=read('2026-10-24/bf16-dual-results.json')
zero=read('2026-10-25/zero3-offload-results.json')
rows=[]
def add(label,seconds,memory,loss,source,cards=2):
 rows.append({'name':label,'cards':cards,'global_batch_size':1,'comparison_steps':[2,3,4],
              'seconds_per_step':statistics.mean(seconds[1:4]),'peak_allocated_gib':memory,
              'validation_loss_after_first4':loss,'evidence':source})
for key,label in [('fp32-cpu-adam','FP32：早期CPU更新对照'),('fp32_full','FP32：初始NPU更新＋CPU状态卸载'),('compressed_b128','FP32计算：ENEC权重＋4-bit m/v，全部驻留NPU')]:
 r=precision['runs'][key];s=[x for x in r['logs'] if x['event']=='step_complete']
 assert r['completed'] and r['data']['train_lengths']==[28,31,31,32]
 memories=[x['memory'] for x in r['logs'] if 'memory' in x]
 add(label,[x['seconds'] for x in s],[max(x[i]['peak_allocated'] for x in memories)/2**30 for i in [0,1]],r['validation_after']['loss'],'2026-10-24/precision-compare-results.json')
for key,label in [('largewindow_npu','FP32：64 MiB窗口＋CPU亲和'),('cpu_master_sync','FP32：CPU主权重＋同步CPUAdam'),('cpu_master_pipeline','FP32：CPU主权重＋更新流水')]:
 r=pipeline['groups'][key]
 add(label,[s['seconds'] for s in r['steps']],r['peak_allocated_gib'],r['validation_after']['loss'],'2026-10-24/fp32-cpu-master-pipeline-results.json')
for r,label in [(mixed,'BF16混合：单卡计算＋CPU FP32主权重/m/v'),(dual,'BF16混合：双卡单进程模型分布')]:
 memory=[max(s['memory'][i]['peak_allocated'] for s in r['steps'])/2**30 for i in range(len(r['physical_devices']))]
 add(label,[s['seconds'] for s in r['steps']],memory,r['validation_after_first4_fp32']['loss'],
     '2026-10-24/bf16-master-training-results.json' if r is mixed else '2026-10-24/bf16-dual-results.json',len(r['physical_devices']))
twenty=[]
for r in [mixed,dual]:
 times=[x['seconds'] for x in r['steps']]
 twenty.append({'cards':len(r['physical_devices']),'steps':20,'global_batch_size':1,'mean_all20_seconds':statistics.mean(times),
                'mean_steps2_20_seconds':statistics.mean(times[1:]),'final_fp32_validation_loss':r['validation_after_fp32']['loss']})
result={'batch1_first4':rows,'batch1_twenty_steps':twenty,'batch2_zero3':{k:zero[k] for k in ['single','dual','throughput_speedup']},
        'notes':['batch1 first4 uses same 28/31/31/32-token examples; exclude first-step warmup',
                 'mixed BF16 memory peak is over entire 20-step run; four-step loss is separate',
                 'batch2 ZeRO-3 uses 8 examples over 4 updates; native BF16 execution validation',
                 'LoRA and failed attempts are separate; no quality ranking across different protocols']}
result['followup_verifications']={}
for filename in ['dual-throughput-verification.json','training-fusion-verification.json']:
 path=root/'evidence/2026-10-25'/filename
 if path.exists():result['followup_verifications'][filename]=json.loads(path.read_text())
(root/'evidence/2026-10-25/training-runs-overview.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 27B训练实测对比','',
 '汇总日期：2026-10-25。仅从既有结果JSON提取，没有启动训练或占用NPU。显存为PyTorch分配峰值，单位GiB，不含驱动。', '',
 '## 每步1个样本：相同前4条训练样本','',
 '全部更新26,895,998,464个文本参数。训练长度28/31/31/32 token，独立8条验证共433个assistant token。速度均取第2–4步平均，排除首步；本表loss统一使用FP32执行验证。', '',
 '| 方案 | 卡数 | 秒/步 | 每卡显存峰值GiB | 4步后验证loss |',
 '|---|---:|---:|---|---:|']
for x in rows:
 mem=' / '.join(f'{n:.2f}' for n in x['peak_allocated_gib'])
 lines.append(f"| {x['name']} | {x['cards']} | {x['seconds_per_step']:.2f} | {mem} | {x['validation_loss_after_first4']:.6f} |")
lines+=['',
 'BF16两组实际各训练20步：上表只取前4步的计时/验证，以匹配旧FP32组；显存峰值取整次20步。单卡BF16的放置、反向实现也改变，因此不把相对FP32的速度差全归因于dtype。压缩组m/v为有损量化，短验证接近不代表更新无损。', '',
 '初始FP32更新到FP32流水：161.76→44.51秒/步，约3.63倍；压缩驻留路线321.88秒/步，容量验证成功，但速度较慢。', '',
 '来源：[原始精度/速度对照](../evidence/2026-10-24/precision-compare-results.json)、[FP32优化序列](../evidence/2026-10-24/fp32-cpu-master-pipeline-results.json)、[单卡BF16](../evidence/2026-10-24/bf16-master-training-results.json)、[双卡BF16](../evidence/2026-10-24/bf16-dual-results.json)。', '',
 '## 每步1个样本：20步完整混合精度运行','',
 '相同20条不同训练样本，长度28–84 token。验证loss取第20步后的统一FP32执行结果，不与上表4步loss当相同训练量比较。', '',
 '| 方案 | 全20步均值秒/步 | 第2–20步均值秒/步 | 20步后验证loss |',
 '|---|---:|---:|---:|']
for x in twenty:
 lines.append(f"| {'单卡' if x['cards']==1 else '双卡单进程模型分布'} | {x['mean_all20_seconds']:.2f} | {x['mean_steps2_20_seconds']:.2f} | {x['final_fp32_validation_loss']:.6f} |")
lines+=['','双卡分布组采用TASK_QUEUE_ENABLE=0；旧单卡未显式记录该变量。双卡分担容量但没有数据并行，排除首步后耗时增加约17.7%。', '',
 '## 每步2个样本：标准ZeRO-3公平对照','',
 '同样8条训练样本、4次全参数更新；单卡累积2个microbatch，双卡每卡1个样本。两组均为原生BF16计算模型、CPU FP32主权重/m/v、相同分块梯度转换、task queue=0。loss是原生BF16执行验证，不能与上表的FP32验证直接当精度排名。', '',
 '| 方案 | 第2–4步均值秒/步 | 训练输入tokens/s | 每卡显存GiB | 最终验证loss | 正确token |',
 '|---|---:|---:|---|---:|---:|']
for key,label in [('single','单卡ZeRO-3，累积2样本'),('dual','双卡ZeRO-3，每卡1样本')]:
 x=zero[key]; mem=' / '.join(f'{n:.2f}' for n in x['peak_allocated_gib']);v=x['validation_after']
 lines.append(f"| {label} | {x['steps2_4_mean_seconds']:.2f} | {x['steps2_4_input_tokens_per_second']:.3f} | {mem} | {v['loss']:.6f} | {v['correct']}/{v['assistant_tokens']} |")
lines+=['',
 f"双卡相对同批量单卡吞吐提升{zero['throughput_speedup']:.3f}倍、每次更新耗时减少33.37%。tokens/s统计提示和回答的全部输入token，是训练吞吐，不是推理生成速度。单卡每步4次allocator cache flush，收益包含容量压力缓解，不能全部归因为纯计算并行。", '',
 '来源：[ZeRO-3完整记录](zero3-offload-20261025/README.md)、[过滤结果JSON](../evidence/2026-10-25/zero3-offload-results.json)。', '',
 '## 历史LoRA与失败记录','',
 '- 单卡BF16 LoRA：只更新约445.6万参数，8次更新/32条训练样本，显存51.56 GiB，验证loss2.1690→1.9409。训练循环169.66秒含前后验证，不能除以8冒充纯训练秒/步，也不能与全参数训练排名。见[LoRA基线](qwen27b-swift-sft/README.md)。',
 '- 初期压缩B128完成两步；B64两次OOM。后续B128四步成功，已列上表，避免把两步与四步混算。',
 '- 自定义BF16反向组因梯度异常、验证恶化在第10步停止，不列为有效性能/精度基准。',
 '- 双卡默认下发队列多次stream/event错误；关闭本进程task queue后完整训练通过。',
 '- 原生单卡ZeRO-3首次反向OOM；有界FP32梯度转换后，单/双卡同配置对照均完成。分块输出位串小测一致，不等于长期整模型训练逐位一致。', '',
 '## 读表结论','',
 '同批量、同数据、同精度配置下，最新标准双进程ZeRO-3已实测约1.50倍吞吐收益。此前纯FP32更新流水也取得明显收益。不能用161.76÷41.64或45.76÷41.64宣称同协议加速，因为批量、计算实现或样本窗口不同。所有结果都是短样本验证，尚不能判定长期质量或硬件性能上限。', '',
 '本次只整理既有数据，未恢复训练或自动Git提交/推送。']
if (root/'evidence/2026-10-25/dual-throughput-verification.json').exists():
 lines+=['','## 后续双卡优化','',
  '本表以上为此前各阶段的冻结结果。后续相同batch=2、8样本4更新的双卡优化达到30.13秒/更新（原41.64秒），验证263/433不变、loss差约−2.38e−7；该收益包含CPU总24→48核及执行配置优化。固定batch32的microbatch/重算/NUMA/预取优化达到34.88秒/32样本。不同批量分开比较，详见[双卡优化报告](dual-throughput-20261025/README.md)和[审计结果](../evidence/2026-10-25/dual-throughput-verification.json)。']
if (root/'evidence/2026-10-25/training-fusion-verification.json').exists():
 lines+=['','## CPU回传流水与社区训练融合','',
  '后续相同batch2、8样本4更新：CPU回传流水28.51秒，叠加固定VeOmni GDN训练融合25.50秒，原30.13秒；batch32相同96次样本曝光/3更新达到27.58秒、86.20输入token/s，原34.88秒。FP32 master/m/v不压缩，融合BF16中间计算与旧FP32分解有数值差异：batch2验证1.499013/266正确token，batch32为1.447144/271正确token，均只代表短跑检查。Profiler计算任务144031→33199，CPUAdam仍约10.7秒。详见[本轮报告](training-fusion-20261025/README.md)及[数值和数据审计](../evidence/2026-10-25/training-fusion-verification.json)。']
(root/'experiments/training-comparison-20261025.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'batch1_rows':len(rows),'twenty_step_runs':len(twenty),'zero3_speedup':zero['throughput_speedup']},ensure_ascii=False))
