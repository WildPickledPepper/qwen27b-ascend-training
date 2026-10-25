"""Export completed rank metrics and data hashes, never weights or text."""
import hashlib,json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
def collect(name,world):
 results=[json.loads((r/name/f'rank{i}-result.json').read_text()) for i in range(world)]
 logs=[[json.loads(line) for line in (r/name/f'rank{i}.jsonl').read_text().splitlines()] for i in range(world)]
 assert all(x['completed'] and len(x['steps'])==4 and x['parameters']==26895998464 and x['fp32_master'] and x['fp32_m_v'] for x in results)
 data=[next(x for x in log if x['event']=='data') for log in logs]
 assert all(x['training_hashes']==data[0]['training_hashes'] and x['validation_hashes']==data[0]['validation_hashes'] for x in data)
 assert not set(data[0]['training_hashes'])&set(data[0]['validation_hashes'])
 assert len(set(data[0]['training_hashes']))==8
 times=[max(x['steps'][i]['seconds'] for x in results) for i in range(4)]
 tokens=[sum(x['steps'][i]['input_tokens'] for x in results) for i in range(4)]
 for result in results:
  assert all(x['global_batch_size']==2 for x in result['steps'])
  assert result['validation_before']==results[0]['validation_before'] and result['validation_after']==results[0]['validation_after']
 assert all(sum(x['steps'][i]['fp32_m_v_bytes'] for x in results)==26895998464*8 for i in range(4))
 return {'world_size':world,'parameters':26895998464,'global_batch_size':2,
         'gradient_accumulation_steps':2//world,'step_seconds':times,'global_input_tokens':tokens,
         'global_training_loss':[sum(x['steps'][i]['weighted_local_loss'] for x in results)/world for i in range(4)],
         'steps2_4_mean_seconds':statistics.mean(times[1:]),
         'steps2_4_input_tokens_per_second':sum(tokens[1:])/sum(times[1:]),
         'training_seconds':sum(times),'peak_allocated_gib':[max(s['peak_allocated'] for s in x['steps'])/2**30 for x in results],
         'peak_reserved_gib':[max(s['peak_reserved'] for s in x['steps'])/2**30 for x in results],
         'fp32_m_v_bytes_by_rank':[x['steps'][-1]['fp32_m_v_bytes'] for x in results],
         'validation_before':results[0]['validation_before'],'validation_after':results[0]['validation_after'],
         'training_hashes':data[0]['training_hashes'],'validation_hashes':data[0]['validation_hashes'],
         'cpu_affinities':[next(x for x in log if x['event']=='begin')['affinity'] for log in logs]}
dual=collect('dual-run2',2);single=collect('single-run2',1)
assert dual['training_hashes']==single['training_hashes'] and dual['validation_hashes']==single['validation_hashes']
assert dual['global_input_tokens']==single['global_input_tokens']
sources=['zero3.json','train_zero3.py','train_zero3_compare.py','train_zero3_bounded.py','bounded_grad_copy.py','launch.sh','run_single_after_dual.py','run_bounded_sequence.py']
manifest=json.loads((r/'bounded-source-manifest.json').read_text())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==v for n,v in manifest.items())
result={'completed':True,'scope':'27B text full-parameter training; native BF16 model; CPU FP32 master and Adam m/v; ZeRO-3',
        'framework':{'deepspeed':'0.19.8','torch':'2.10.0+cpu','torch_npu':'2.10.0.post4','transformers':'5.14.1'},
        'dual':dual,'single':single,'throughput_speedup':dual['steps2_4_input_tokens_per_second']/single['steps2_4_input_tokens_per_second'],
        'same_global_batch_and_examples':True,'task_queue_enable':'0','checkpoint_saved':False,
        'bounded_grad_copy_probe':json.loads((r/'bounded-copy-probe.json').read_text()),
        'original_dual_without_adapter':collect('dual-run1',2),
        'original_single_failure':'First backward OOM: 4.74 GiB full-parameter FP32 gradient cast with 3.16 GiB free; no full update completed',
        'single_allocator_pressure':'Training log reports four allocator cache flushes per update; capacity effects are part of the observed speed difference',
        'source_sha256_lf':{n:hashlib.sha256((r/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in sources},
        'tiny_two_rank_probe':[json.loads((r/'tiny-run2'/f'rank{i}-result.json').read_text()) for i in [0,1]],
        'limits':['four updates, short sequences only','native BF16 operation path differs from previous custom mixed harness','no complete new checkpoint saved','not a long-term quality or maximum throughput claim']}
(r/'zero3-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
