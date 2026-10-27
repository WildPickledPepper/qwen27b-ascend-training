"""Measured phase bounds plus explicitly illustrative execution mechanisms."""
from pathlib import Path
import json, html, math
from PIL import Image, ImageDraw, ImageFont

out = Path(__file__).resolve().parent
root = out.parents[1]
data = json.loads((root / 'evidence/2026-10-27/sql-cache-tail-results.json').read_text(encoding='utf-8'))
rows = []
for rank, r in enumerate(data['runs']['cache4-8']['rank_results']):
    s = r['steps'][1]
    f = s['forward_backward_and_drain_seconds']
    n = f + s['global_norm_seconds']
    e = s['seconds']
    rows.append(dict(card=rank+3, fb=f, norm=n, end=e, barrier=e-s['tail_seconds']['end_barrier']))

W, H = 1900, 1940
im = Image.new('RGB', (W, H), '#101722')
dr = ImageDraw.Draw(im)
svg = []

def rect(x, y, w, h, c):
    dr.rectangle((x, y, x+w, y+h), fill=c)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{c}"/>')

def text(x, y, s, n=25, c='#e4edf9'):
    dr.text((x, y), s, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', n), fill=c)
    svg.append(f'<text x="{x}" y="{y+n*.93}" font-size="{n}" fill="{c}">{html.escape(s)}</text>')

def line(x, y, xx, yy, c='#48576b', w=2):
    dr.line((x, y, xx, yy), fill=c, width=w)
    svg.append(f'<line x1="{x}" y1="{y}" x2="{xx}" y2="{yy}" stroke="{c}" stroke-width="{w}"/>')

def arrow(x, y, xx, yy, c='#62dcc0'):
    line(x, y, xx, yy, c, 3)
    a = math.atan2(yy-y, xx-x)
    for d in [-.6, .6]:
        line(xx, yy, xx-14*math.cos(a+d), yy-14*math.sin(a+d), c, 3)

def bar(x, y, w, label, c, n=23):
    rect(x, y, w, 49, c)
    if label:
        text(x+9, y+9, label, n, '#ffffff')

def panel(y, h, title, subtitle):
    rect(28, y, 1844, h, '#192332')
    text(53, y+17, title, 30)
    text(53, y+65, subtitle, 22, '#a6b9d1')

C = dict(blue='#2477a4', yellow='#b67a18', purple='#7958b2', green='#168f71', red='#bf455b', copy='#bf742d', gray='#445368')
text(48, 24, '训练多泳道：CPU梯度累加到底在哪？', 42)
text(50, 91, '绿色 = 把梯度加起来；紫色 = 用梯度更新权重。这是两个不同过程。', 28, '#88e4cc')
panel(150, 425, 'A｜实测：同一次更新的两张卡', 'global256 · micro16 · 4/8GiB梯度缓存 · 第2次更新 · 只画已有计时边界，未虚构线程事件')
xx = lambda t: 250+t/78*1560
for t in range(0, 78, 10):
    line(xx(t), 253, xx(t), 425, '#334155', 1)
    text(xx(t)-12, 430, str(t)+'s', 19, '#9fb1c8')
for i, r in enumerate(rows):
    y = 269+i*84
    text(60, y+10, str(r['card'])+'号卡主阶段', 25)
    for a, b, label, c in [(0, r['fb'], '前后向＋梯度搬运＋CPU累加＋排空', C['blue']), (r['fb'], r['norm'], '', C['yellow']), (r['norm'], r['barrier'], 'CPUAdam＋回传', C['purple']), (r['barrier'], r['end'], '', C['red'])]:
        bar(xx(a), y, max(.4, xx(b)-xx(a)), label, c, 21)
    text(xx(r['fb'])-42, y-28, f"{r['fb']:.2f}s", 19, '#ffd48d')
text(62, 478, '蓝色内部：3号主线程等空槽累计35.91s；4号26.46s。没有逐事件位置，不能铺成连续红条。', 24, '#f1a9b3')
text(62, 519, '黄色：本地范数＋同步；63.71s后两卡开始更新。末尾这一步3号等4号约0.396s。', 23, '#b5c5da')

panel(600, 765, 'B｜放大蓝色内部：并发与“谁等谁”的机制', '以下是代码对应的执行示意，不是实测事件时序；为看清堵塞只画2个接收槽，实际每卡有16个槽')
sx = lambda t: 270+t/110*1530
ys = [735, 842, 949, 1056, 1163]
for y, label in zip(ys, ['训练主线程', 'NPU执行流', 'CPU累加线程', 'CPU接收槽A', 'CPU接收槽B']):
    text(53, y+10, label, 25)
    line(270, y+64, 1800, y+64, '#2c3b50', 1)
for t in [0, 20, 40, 60, 80, 100]:
    line(sx(t), 710, sx(t), 1230, '#26384c', 1)

def b(t, u, row, label, c, n=21):
    bar(sx(t), ys[row], sx(u)-sx(t), label, C[c], n)

b(0,10,0,'提交A','blue',18)
b(10,30,0,'继续反向/提交B','blue',19)
b(30,45,0,'继续反向','blue',19)
b(45,65,0,'槽满：等空槽','red',20)
b(65,78,0,'提交C','blue',19)
b(78,110,0,'继续训练','blue',21)
b(0,10,1,'计算A','blue',18)
b(10,20,1,'D2H A','copy',18)
b(20,30,1,'计算B','blue',18)
b(30,40,1,'D2H B','copy',18)
b(40,50,1,'已排队任务','blue',17)
b(50,65,1,'队列耗尽可空闲','gray',16)
b(65,78,1,'D2H C','copy',19)
b(78,110,1,'后续计算','blue',21)
b(10,20,2,'等A传完','gray',16)
b(20,65,2,'CPU：把A加进累计梯度','green',23)
b(65,110,2,'CPU：把B加进累计梯度','green',23)
b(10,65,3,'A正在接收/累加，槽不能复用','gray',23)
b(65,110,3,'槽A腾出，接收新的C','copy',23)
b(30,110,4,'B已排队，等CPU处理完成后才归还槽','gray',23)
x = sx(65)
arrow(x, ys[2]+49, x, ys[0]+50)
text(x+12, ys[1]+63, 'A累加完 → 归还槽 → 主线程继续', 21, '#85e1c9')
arrow(sx(20), ys[1]+49, sx(20), ys[2])
text(sx(22), ys[2]+57, '绿色与NPU后续任务可并行；不能把两条泳道的时间相加。', 22, '#85e1c9')
text(53, 1260, '当前没有独立D2H流：橙色搬运与蓝色NPU计算在同一执行流依次排队；绿色CPU工作另在线程里进行。', 22, '#b6c6dc')
text(53, 1300, '4号8GiB驻留梯度在NPU内累加，16个micro结束后才卸载一次；这部分不逐micro进入上方CPU队列。', 22, '#b6c6dc')

panel(1390, 385, 'C｜蓝色结束前先排空，之后裁剪，再进入紫色', '排空 = 等NPU驻留梯度卸载、CPU队列累加都完成；不是丢弃梯度')
for x,w,label,c in [(52,160,'反向全部完成',C['blue']),(395,280,'梯度搬运/累加收尾',C['green']),(945,260,'全局范数与裁剪',C['yellow']),(1460,350,'CPUAdam更新＋权重回传',C['purple'])]:
    bar(x, 1515, w, label, c, 21)
for a,b_ in [(212,395),(675,945),(1205,1460)]:
    arrow(a+10,1540,b_-12,1540,'#8da5c5')
text(53, 1612, '我们一直优化的CPU梯度累加 = B中的绿色，属于A的蓝色大阶段；紫色不是那个累加。', 26, '#88e4cc')
text(53, 1665, '紫色内部已有部分并发：更新下一组CPU参数时，可回传上一组新权重；不是所有小操作都串行。', 23, '#c9b9ee')
text(53, 1710, '主要堵塞链：CPU归还接收槽不够快 → 主线程等槽 → 后续任务下发受阻；何时NPU空闲要看设备trace。', 22, '#f2b5bc')
text(48, 1810, '数据来源：sql-cache-tail-results.json，4/8GiB组第2步；B只说明机制，不用于计算任何性能收益。', 23, '#9fb1c8')
text(48, 1856, '不要把35.91s理解为连续停机，也不要把黄色整段理解为纯等待。大阶段关系与内部线程并发要分开看。', 23, '#9fb1c8')
im.save(out/'detailed-training-swimlanes.png')
svgstr = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="详细训练多泳道时间线"><rect width="{W}" height="{H}" fill="#101722"/>'+''.join(svg)+'</svg>'
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>训练多泳道</title><style>body{margin:0;background:#101722;color:#e4edf9;font-family:"Microsoft YaHei",sans-serif}nav{position:sticky;top:0;padding:12px 24px;background:#233247;display:flex;gap:12px;align-items:center;z-index:2}button{padding:9px 16px;background:#30455f;color:white;border:1px solid #67809e;border-radius:6px;cursor:pointer}main{overflow:auto}svg{display:block;width:100%;min-width:1100px;font-family:"Microsoft YaHei",sans-serif}rect:hover{filter:brightness(1.12)}</style><nav><button onclick="zoom(1)">适应窗口</button><button onclick="zoom(1.5)">放大1.5倍</button><button onclick="go(150)">A 实测阶段</button><button onclick="go(600)">B 蓝色内部并发</button><button onclick="go(1390)">C 收尾与更新</button><span>A：实测边界　B：机制示意，非设备trace</span></nav><main>SVG</main><script>function zoom(n){document.querySelector('svg').style.width=(n*100)+'%'}function go(y){const s=document.querySelector('svg');window.scrollTo({top:s.getBoundingClientRect().top+window.scrollY+y*s.getBoundingClientRect().width/1900-65,behavior:'smooth'})}</script></html>'''.replace('SVG',svgstr)
(out/'detailed-training-swimlanes.html').write_text(page, encoding='utf-8')
print(out/'detailed-training-swimlanes.png')
