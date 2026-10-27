"""Render the actual second update; no artificial decomposition of slot waits."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
out=Path(__file__).resolve().parent;root=out.parents[1]
d=json.loads((root/'evidence/2026-10-27/sql-cache-tail-results.json').read_text(encoding='utf-8'))
rows=[]
for rank,r in enumerate(d['runs']['cache4-8']['rank_results']):
 s=r['steps'][1];f=s['forward_backward_and_drain_seconds'];n=f+s['global_norm_seconds'];e=s['seconds'];b=e-s['tail_seconds']['end_barrier']
 rows.append(dict(card=rank+3,fb=f,norm=n,barrier=b,end=e,segments=[(0,f,'前后向＋梯度排空','#3b82b0'),(f,n,'本地范数＋全局同步','#e5a329'),(n,b,'CPU更新＋转换＋权重回传','#8262bc'),(b,e,'末尾等待另一张卡','#df5866')]))
im=Image.new('RGB',(1680,1070),'#f4f7fb');draw=ImageDraw.Draw(im)
def text(x,y,s,n=24,c='#233247'):draw.text((x,y),s,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n),fill=c)
def box(x,y,w,h,c):draw.rounded_rectangle((x,y,x+w,y+h),radius=10,fill=c)
text(56,30,'谁在等谁：双卡训练真实时间线',38);text(56,88,'4/8 GiB 梯度缓存 · 第 2 次更新 · 256 个样本 · 本步耗时 76.75 秒',23,'#57677c')
for i,(label,c) in enumerate([('前后向＋梯度排空','#3b82b0'),('本地范数＋同步','#e5a329'),('CPU更新＋权重回传','#8262bc'),('末尾卡间等待','#df5866')]):
 x=58+i*398;box(x,137,18,18,c);text(x+29,130,label,21)
def panel(y,a,b,title,ys,labels):
 box(36,y,1608,300,'#ffffff');text(60,y+15,title,26)
 def xx(t):return 200+(t-a)/(b-a)*1380
 ticks=list(range(0,81,10)) if a==0 else [58,60,62,64,66,68,70,72,74,76,77]
 for t in ticks:
  if a<=t<=b:
   x=xx(t);draw.line((x,y+65,x,y+238),fill='#dde4ed',width=1);text(x-15,y+243,str(t),18,'#65758b')
 for i,r in enumerate(rows):
  yy=y+ys[i];text(62,yy+11,str(r['card'])+'号卡',26)
  for st,en,label,c in r['segments']:
   st=max(st,a);en=min(en,b)
   if en<=st:continue
   xa,xb=xx(st),xx(en);draw.rectangle((xa,yy,xb,yy+52),fill=c)
   if xb-xa>240:text(xa+12,yy+12,label,20,'white')
  if labels:
   text(xx(r['fb'])-55,yy-29,format(r['fb'],'.2f')+'s',19,'#2a658e');draw.line((xx(r['norm']),yy-4,xx(r['norm']),yy+58),fill='#4c426e',width=2)
 if labels:text(xx(rows[0]['norm'])+12,y+218,'63.71s：两卡对齐后开始更新',20,'#554171')
 return xx
panel(181,0,78,'① 全程：先看两个阶段谁先完成',[87,168],False)
xx=panel(501,58,77.3,'② 放大最后 19 秒：看同步点与末尾等待',[84,164],True)
x=xx(rows[0]['barrier']);draw.line((x,637,x,654),fill='#df5866',width=2)
text(930,781,'3号先完成更新，末尾等4号约 0.396 秒',21,'#b13e4e')
box(36,828,1608,207,'#eaf0f8');text(60,845,'读图重点',25)
text(60,887,'• 4号在59.70s完成计算与排空；3号在62.38s完成。领先约2.68s，在范数同步段消失。',23)
text(60,929,'• 主线程等CPU接收槽：3号累计35.91s，4号26.46s；分散在蓝色阶段内，未画成连续空闲。',23)
text(60,975,'黄色包含本地计算与同步，不能全部算等待；接收槽等待与计算重叠，也不能相加。',21,'#5e6e83')
im.save(out/'cache-wait-timeline.png')
