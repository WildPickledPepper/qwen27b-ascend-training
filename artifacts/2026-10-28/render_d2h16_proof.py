"""Render one measured starvation window; native DMA duration, no schematic timing."""
from pathlib import Path
import json,html,math
from PIL import Image,ImageDraw,ImageFont
out=Path(__file__).resolve().parent;root=out.parents[1]
d=json.loads((root/'evidence/2026-10-28/d2h16-03-figure-case.json').read_text(encoding='utf-8'));c=d['case']
W,H=1880,1160;im=Image.new('RGB',(W,H),'#f3f6fb');dr=ImageDraw.Draw(im);sv=[]
def box(x,y,w,h,color):
 dr.rectangle((x,y,x+w,y+h),fill=color);sv.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
def text(x,y,s,n=24,color='#26354a'):
 dr.text((x,y),s,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n),fill=color);sv.append(f'<text x="{x}" y="{y+n*.93}" font-size="{n}" fill="{color}">{html.escape(s)}</text>')
def line(x,y,x2,y2,col='#b3c0d2',w=2):
 dr.line((x,y,x2,y2),fill=col,width=w);sv.append(f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}"/>')
def dot(x,y,col):
 dr.ellipse((x-6,y-6,x+6,y+6),fill=col);sv.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{col}"/>')
text(40,26,'固定16槽：CPU为什么会断粮？',39)
text(42,82,'真实Level2 msprof记录 · 3号卡 · 每槽128MiB · 0/3双卡训练 · 不是机制示意图',25,'#596a80')
box(28,138,1824,744,'#ffffff')
a,b=18.8,20.36;x0,x1=250,1800
xx=lambda t:x0+(t-a)/(b-a)*(x1-x0)
for t in [18.8,19,19.2,19.4,19.6,19.8,20,20.2,20.35]:
 line(xx(t),178,xx(t),835,'#e1e7ef',1);text(xx(t)-24,842,f'{t:.2f}s',18,'#69798e')
labels=[(208,'CPU累加线程'),(326,'卡间通信任务'),(444,'NPU计算任务'),(582,'主机下发D2H'),(686,'实际DMA'),(792,'缓冲槽快照')]
for y,s in labels:text(48,y+8,s,25)
w0,w1=c['cpu_get_wait_start_seconds'],c['cpu_get_return_seconds']
box(xx(w0),208,xx(w1)-xx(w0),52,'#ce566a');text(xx(w0)+18,219,'等待下一份梯度：约1.341秒',25,'white')
for q in c['communications_before_source_ready']:
 l=max(a,q['start_seconds']);u=min(b,q['end_seconds'])
 if u>l:
  box(xx(l),326,max(1,xx(u)-xx(l)),48,'#8d77b5')
  if u-l>.2:text(xx(l)+15,335,'HCCL同步 / 等待对端',21,'white')
text(xx(19.5),383,'通信覆盖约0.532秒，包含等待，不是纯数据传输',21,'#675580')
for st,en,name in c['compute_intervals_relative_step']:
 l=max(w0,st);u=min(c['source_ready_seconds'],en)
 if u>l:box(xx(l),444,max(.6,xx(u)-xx(l)),48,'#2885b0')
text(xx(19.49),504,'下一micro前向＋输出头反向，计算覆盖约0.792秒',21,'#216787')
api=c['runtime_api_start_seconds'];dot(xx(api),605,'#c68c20');line(xx(api),582,xx(api),632,'#c68c20',3)
text(xx(api)-350,638,f'D2H已在{api:.3f}s提交，仍需等待梯度就绪',22,'#886315')
ready=c['source_ready_seconds'];x=xx(ready)
for y in range(186,826,12):line(x,y,x,y+6,'#546984',2)
text(xx(19.93),154,'约20.264s：梯度就绪',22,'#3e5778')
dot(xx(c['native_copy_start_seconds']),710,'#ed8529')
text(xx(18.97),698,'10KiB梯度：实际DMA仅约2.5微秒',24,'#ad5f19')
text(xx(18.97),735,'橙点只标位置；真实宽度在此比例下不可见',19,'#7c6c5c')
snap=c['snapshot'];dot(xx(snap['host']),815,'#238c71');text(xx(snap['host'])+16,791,'此刻：16个空槽，0份就绪数据',24,'#267c65')
box(28,912,1824,202,'#e7eef7')
text(50,928,'结论：等的是下一份梯度产出，不是DMA把10KiB传了1.34秒。',29)
text(50,982,'CPU已处理完上一批 → 等卡间同步 → NPU算下一micro的前向/输出头反向 → 梯度就绪后立即搬。',23)
text(50,1029,'16个micro之间出现15次相似断粮；3号卡15次快照均为16空槽。缓冲区不能装尚未产出的梯度。',23)
text(50,1073,'计算任务覆盖与通信可能重叠；数字不机械相加。跨时钟对齐为近似，DMA时长直接取硬件TASK记录。',20,'#61738b')
im.save(out/'d2h16-measured-starvation.png')
svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#f3f6fb"/>'+''.join(sv)+'</svg>'
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>固定16槽：真实搬运等待</title><style>body{margin:0;background:#f3f6fb;font-family:"Microsoft YaHei",sans-serif}nav{position:sticky;top:0;background:#e7eef7;padding:12px;z-index:2}button{padding:8px 18px;margin-right:12px;border:1px solid #9fb1c8;background:white;border-radius:6px;cursor:pointer}main{overflow:auto}svg{width:100%;min-width:1100px;display:block;font-family:"Microsoft YaHei",sans-serif}</style><nav><button onclick="z(1)">适应窗口</button><button onclick="z(1.5)">放大1.5倍</button><button onclick="z(2)">放大2倍</button>真实设备任务时间线；橙点是2.5微秒DMA的位置，不按可见宽度表示。</nav><main>SVG</main><script>function z(n){document.querySelector('svg').style.width=(n*100)+'%'}</script></html>'''.replace('SVG',svg)
(out/'d2h16-measured-starvation.html').write_text(page,encoding='utf-8')
