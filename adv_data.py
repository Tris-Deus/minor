import numpy as np
import pandas as pd
import random
from collections import Counter


dataset=pd.read_csv("new_Stress.csv")
good=[0,1]
ok=[2,3]
bad=[3,4]
eve=[0,1,2,3,4]

upset=[]
control=[]
nerv=[]
conf=[]
way=[]
cope=[]
irr=[]
top=[]
anger=[]
diff=[]

sr=dataset.iloc[:,0].values
rr=dataset.iloc[:,1].values
t=dataset.iloc[:,2].values
lm=dataset.iloc[:,3].values
bo=dataset.iloc[:,4].values
rem=dataset.iloc[:,5].values
sle=dataset.iloc[:,6].values
hr=dataset.iloc[:,7].values
emo=dataset.iloc[:,8].values
sl=dataset.iloc[:,9].values

for i in range(len(sl)):
    if t[i]>99 and bo[i]<95 and emo[i] in [0,1,5] and sl[i]>=3:
        upset.append(random.choice(bad))
        control.append(random.choice(bad))
        nerv.append(random.choice(bad))
        conf.append(random.choice(bad))
        way.append(random.choice(good))
        cope.append(random.choice(bad))
        irr.append(random.choice(good))
        top.append(random.choice(good))
        anger.append(random.choice(bad))
        diff.append(random.choice(ok))
    elif sle[i]<7 and sl[i]>=2:
        upset.append(random.choice(bad))
        control.append(random.choice(ok))
        nerv.append(random.choice(ok))
        conf.append(random.choice(ok))
        way.append(random.choice(ok))
        cope.append(random.choice(ok))
        irr.append(random.choice(bad))
        top.append(random.choice(ok))
        anger.append(random.choice(bad))
        diff.append(random.choice(ok))
    elif t[i]>99 and sl[i]<=1:
        upset.append(random.choice(good))
        control.append(random.choice(good))
        nerv.append(random.choice(good))
        conf.append(random.choice(ok))
        way.append(random.choice(bad))
        cope.append(random.choice(good))
        irr.append(random.choice(bad))
        top.append(random.choice(bad))
        anger.append(random.choice(good))
        diff.append(random.choice(ok)) 
    elif t[i]>99 and bo[i]>=94.75 and (rem[i]<76.25 or sle<7):
        upset.append(random.choice(ok))
        control.append(random.choice(bad))
        nerv.append(random.choice(bad))
        conf.append(random.choice(good))
        way.append(random.choice(ok))
        cope.append(random.choice(bad))
        irr.append(random.choice(bad))
        top.append(random.choice(good))
        anger.append(random.choice(bad))
        diff.append(random.choice(bad))
    elif t[i]<99 and t[i]>96 and rem[i]<=76.25 and sl[i]<=1:
        upset.append(random.choice(good))
        control.append(random.choice(good))
        nerv.append(random.choice(good))
        conf.append(random.choice(bad))
        way.append(random.choice(ok))
        cope.append(random.choice(good))
        irr.append(random.choice(bad))
        top.append(random.choice(ok))
        anger.append(random.choice(good))
        diff.append(random.choice(good))
    elif sl[i]>2:
        upset.append(random.choice(good))
        control.append(random.choice(ok))
        nerv.append(random.choice(bad))
        conf.append(random.choice(ok))
        way.append(random.choice(bad))
        cope.append(random.choice(bad))
        irr.append(random.choice(good))
        top.append(random.choice(ok))
        anger.append(random.choice(bad))
        diff.append(random.choice(bad))
    else:
        upset.append(random.choice(eve))
        control.append(random.choice(eve))
        nerv.append(random.choice(eve))
        conf.append(random.choice(eve))
        way.append(random.choice(eve))
        cope.append(random.choice(eve))
        irr.append(random.choice(eve))
        top.append(random.choice(eve))
        anger.append(random.choice(eve))
        diff.append(random.choice(eve))

dataset.insert(loc=0,column='diff',value=diff)
dataset.insert(loc=0,column='anger',value=anger)
dataset.insert(loc=0,column='top',value=top)
dataset.insert(loc=0,column='irr',value=irr)
dataset.insert(loc=0,column='cope',value=cope)
dataset.insert(loc=0,column='way',value=way)
dataset.insert(loc=0,column='conf',value=conf)
dataset.insert(loc=0,column='nerv',value=nerv)
dataset.insert(loc=0,column='control',value=control)
dataset.insert(loc=0,column='upset',value=upset)

dataset.to_csv('adv_stress.csv')
