import os
os.environ['DEVELOPMENT']='True'

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement

torch.set_num_threads(1)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def make_soil_crop():
    s=Soil('custom',cn=46,rew=7)
    s.add_layer(0.1,0.268,0.434,0.620,47.2,100)
    s.add_layer(0.1,0.279,0.439,0.618,14.3,100)
    s.add_layer(0.1,0.301,0.480,0.618,9.1,100)
    s.add_layer(0.1,0.293,0.500,0.611,8.0,100)
    c=Crop('custom',CropType=1,PlantMethod=0,CalendarType=1,SwitchGDD=0,
           planting_date='10/01',harvest_date='10/30',EmergenceCD=6,MaxRootingCD=22,SenescenceCD=23,MaturityCD=30,
           HIstart=6,FloweringCD=0,YldFormCD=20,Tbase=10.0,Tupp=30.0,Zmin=0.30,Zmax=0.30,fshape_r=15,SxTopQ=0.048,SxBotQ=0.012,
           CCx=0.85,CGC_CD=0.4,CDC_CD=0.08,Kcb=1.10,WP=17,WPy=100,fsink=1,HI0=65,
           fshape_w1=2.5,fshape_w2=3.0,fshape_w3=3.0,fshape_w4=1.0,
           p_up1=0.25,p_up2=0.50,p_up3=0.85,p_up4=0.90,Tmin_up=8,Tmax_up=40,Aer=5,LagAer=3,
           PlantPop=160000,SeedSize=15,p_lo1=0.20,p_lo2=0.25,p_lo3=0.30,p_lo4=0.45,Determinant=1,HIstartCD=15,YldWC=90)
    wc=InitialWaterContent(wc_type='Pct',value=[40])
    return s,c,wc

def drop_feb29(df):
    m=(df['Date'].dt.month==2)&(df['Date'].dt.day==29)
    return df.loc[~m].reset_index(drop=True)

def read_climate(path):
    df=pd.read_excel(path)
    df['Date']=pd.to_datetime(df['Date'])
    df=df.sort_values('Date').reset_index(drop=True)
    return drop_feb29(df)

class OnlineZ:
    def __init__(self, dim, eps=1e-8):
        self.dim=dim; self.eps=eps
        self.n=0; self.mean=np.zeros(dim,dtype=np.float64); self.M2=np.zeros(dim,dtype=np.float64)
    def update(self, x):
        x=np.asarray(x,dtype=np.float64)
        self.n+=1
        delta=x-self.mean
        self.mean+=delta/self.n
        delta2=x-self.mean
        self.M2+=delta*delta2
    def transform(self, x):
        var=np.where(self.n>1,self.M2/(self.n-1),np.ones_like(self.M2))
        std=np.sqrt(np.maximum(var,self.eps))
        return ((x-self.mean)/std).astype(np.float32)
    def state_dict(self):
        return {'dim':self.dim,'eps':self.eps,'n':self.n,'mean':self.mean,'M2':self.M2}
    def load_state_dict(self, st):
        self.dim=st['dim']; self.eps=st['eps']; self.n=st['n']; self.mean=st['mean']; self.M2=st['M2']

class DQN(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc1=nn.Linear(inp,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,out)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99, eps0=1.0, eps_decay=0.98, eps_min=0.05, capacity=5000):
        self.q=DQN(state_dim,n_actions).to(device)
        self.opt=optim.Adam(self.q.parameters(),lr=lr)
        self.crit=nn.MSELoss()
        self.mem=deque(maxlen=capacity)
        self.gamma=gamma
        self.eps=eps0; self.eps_decay=eps_decay; self.eps_min=eps_min
        self.n_actions=n_actions
    def act(self, s):
        if np.random.rand()<=self.eps: return np.random.randint(self.n_actions)
        with torch.no_grad():
            q=self.q(torch.as_tensor(s,dtype=torch.float32,device=device).unsqueeze(0))
            return int(torch.argmax(q,dim=1).item())
    def store(self, s,a,r,sn,d):
        self.mem.append((s,a,r,sn,d))
    def train_step(self, batch=64):
        if len(self.mem)<batch: return
        idx=np.random.choice(len(self.mem),batch,replace=False)
        ss,aa,rr,sn,dd=zip(*[self.mem[i] for i in idx])
        ss=torch.as_tensor(np.stack(ss),dtype=torch.float32,device=device)
        aa=torch.as_tensor(aa,dtype=torch.int64,device=device)
        rr=torch.as_tensor(rr,dtype=torch.float32,device=device)
        sn=torch.as_tensor(np.stack(sn),dtype=torch.float32,device=device)
        dd=torch.as_tensor(dd,dtype=torch.float32,device=device)
        qsa=self.q(ss).gather(1,aa.view(-1,1)).squeeze(1)
        with torch.no_grad():
            qmax=self.q(sn).max(1).values
            tgt=rr + self.gamma*qmax*(1.0-dd)
        loss=self.crit(qsa,tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

def run_aquacrop_slice(weather_df, schedule_depth):
    s,c,wc=make_soil_crop()
    sim_start=weather_df['Date'].iloc[0].strftime('%Y/%m/%d')
    sim_end=weather_df['Date'].iloc[-1].strftime('%Y/%m/%d')
    sched=pd.DataFrame({'Date':weather_df['Date'].iloc[:len(schedule_depth)].to_list(),
                        'Depth':np.asarray(schedule_depth,dtype=float)})
    model=AquaCropModel(sim_start,sim_end,weather_df,s,c,IrrigationManagement(3,Schedule=sched),wc)
    model.run_model(till_termination=True)
    out=model._outputs
    fs=getattr(out,'final_stats',None)
    if fs is not None and len(fs)>0 and 'Dry yield (tonne/ha)' in fs.columns:
        return float(fs['Dry yield (tonne/ha)'].values[0])
    return 0.0

def train_dql(climate_path, episodes=8000, episode_len=30, batch=64, save_each=100):
    df=read_climate(climate_path)
    cols=['MinTemp','MaxTemp','Precipitation','ReferenceET']
    actions=np.arange(0,21,dtype=float)
    state_dim=4; agent=Agent(state_dim,len(actions))
    scaler=OnlineZ(state_dim)
    Y_star=20.0; wY=1.0; lambda_R=0.01; eps_phi=1e-6
    results=[]; summaries=[]
    for ep in range(1,episodes+1):
        start=np.random.randint(0, max(1,len(df)-episode_len))
        sl=df.iloc[start:start+episode_len].reset_index(drop=True)
        sched=[]
        ep_ret=0.0
        for t in range(len(sl)):
            s_raw=sl.loc[t,cols].to_numpy(dtype=np.float64)
            scaler.update(s_raw)
            s=scaler.transform(s_raw)
            a_idx=agent.act(s)
            I_t=actions[a_idx]
            sched.append(I_t)
            Y_t=run_aquacrop_slice(sl[['Date']+cols],sched)
            P=float(sl.loc[t,'Precipitation']); E=float(sl.loc[t,'ReferenceET'])
            phi=max(0.0,(P-E)/(E+eps_phi))
            r=wY*(Y_t/Y_star) - lambda_R*I_t*phi
            if t+1<len(sl):
                s1_raw=sl.loc[t+1,cols].to_numpy(dtype=np.float64)
                s1=scaler.transform(s1_raw)
            else:
                s1=np.zeros_like(s,dtype=np.float32)
            d=float(t==len(sl)-1)
            agent.store(s,a_idx,r,s1,d)
            agent.train_step(batch=batch)
            ep_ret+=r
            results.append([ep,t+1,I_t,r,Y_t,phi,P,E])
        summaries.append([ep,ep_ret])
        if ep%save_each==0:
            torch.save({'q':agent.q.state_dict(),
                        'scaler':scaler.state_dict(),
                        'eps':agent.eps}, f"dql_ep{ep}.pth")
        agent.eps=max(agent.eps*agent.eps_decay, agent.eps_min)
    res=pd.DataFrame(results,columns=['episode','t','I_mm','reward','Y_t','phi','P','E'])
    summ=pd.DataFrame(summaries,columns=['episode','return'])
    res.to_excel("resultados_treinamento_rl.xlsx",index=False)
    summ.to_excel("resumo_treinamento_rl.xlsx",index=False)
    return agent, scaler, res, summ

def load_agent(state_dim, n_actions, path):
    ckpt=torch.load(path,map_location=device)
    agent=Agent(state_dim,n_actions)
    agent.q.load_state_dict(ckpt['q'])
    if 'eps' in ckpt: agent.eps=float(ckpt['eps'])
    scaler=OnlineZ(state_dim); scaler.load_state_dict(ckpt['scaler'])
    agent.q.eval()
    return agent, scaler

def predict_irrigation(agent, scaler, min_temp, max_temp, precipitation, reference_et):
    x=np.array([min_temp,max_temp,precipitation,reference_et],dtype=np.float64)
    z=scaler.transform(x)
    with torch.no_grad():
        q=agent.q(torch.as_tensor(z,dtype=torch.float32,device=device).unsqueeze(0))
        a=int(torch.argmax(q,dim=1).item())
    return float(a)
