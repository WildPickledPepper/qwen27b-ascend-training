import bisect
def merge(intervals):
 out=[]
 for s,e,*_ in sorted(intervals):
  if e<=s:continue
  if out and s<=out[-1][1]:out[-1][1]=max(e,out[-1][1])
  else:out.append([s,e])
 return out
def difference(intervals,remove):
 out=[];j=0
 for s,e in intervals:
  while j<len(remove) and remove[j][1]<=s:j+=1
  k=j;cursor=s
  while k<len(remove) and remove[k][0]<e:
   a,b=remove[k]
   if a>cursor:out.append([cursor,min(a,e)])
   cursor=max(cursor,b)
   if cursor>=e:break
   k+=1
  if cursor<e:out.append([cursor,e])
 return out
class Measure:
 def __init__(self,intervals):
  self.x=merge(intervals);self.s=[x[0] for x in self.x];self.e=[x[1] for x in self.x];self.prefix=[0]
  for s,e in self.x:self.prefix.append(self.prefix[-1]+e-s)
 def before(self,t):
  i=bisect.bisect_right(self.e,t);v=self.prefix[i]
  if i<len(self.x):v+=max(0,t-self.s[i])
  return v
 def inside(self,s,e):return self.before(e)-self.before(s)
 def union_inside(self,intervals):return sum(self.inside(s,e) for s,e in merge(intervals))
