# -*- coding: utf-8 -*-
"""N7 v2: ABI 深度学习三模型（npjDM 对齐，干净实现）
张量: 每检查点 → 尾随 48h 小时桶 (48, n_vars×2) 值+掩码
分箱: 每变量每患者预建小时数组 → 切片取窗（不用逐行循环）
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate
gate(__file__)

SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)
R = "E:/TBI subtype/09_tbi_aki/results"; H = 48; DEVICE = "cpu"

# ============================ 加载 ============================
F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
vit = pd.read_parquet(f"{R}/n4_abi_vitals.parquet")
labs = pd.read_parquet(f"{R}/n4_abi_labs.parquet").dropna(subset=["hr"])
uo_rows = pd.read_parquet(f"{R}/n2_m4_abi_uo_rows.parquet")
import duckdb
con = duckdb.connect()
adm = con.execute(f"""
SELECT c.stay_id, c.subject_id, a.admittime, p.anchor_year
FROM read_parquet('{R}/n1_m4_abi_cohort.parquet') c
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/admissions.csv.gz') a ON c.hadm_id=a.hadm_id
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz') p ON c.subject_id=p.subject_id
""").df()
pat2 = pd.read_csv("E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz", usecols=["subject_id","anchor_year_group"])
_g = pat2["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
pat2["group_mid"] = (_g[0].astype(int)+_g[1].astype(int))/2.0
coh = adm.merge(pat2[["subject_id","group_mid"]], on="subject_id", how="left")
coh["year"] = (coh.group_mid + (coh.admittime.dt.year - coh.anchor_year)).round().astype(int)
F = F.merge(coh[["stay_id","year"]], on="stay_id", how="left")

# ============================ 变量定义（12 个时序 + 静态从 F 直取） ============================
TS_VARS = ["hr_rate","sbp","spo2","temp_c","rr","cr","na","k","glu","bun","hco3","uo"]
NV_TS = len(TS_VARS)

# 预建小时数组 per (patient, var)
VIT_MAP = {220045:"hr_rate",220179:"sbp",220277:"spo2",223762:"temp_c",220210:"rr"}
LAB_MAP = {50912:"cr",50983:"na",50971:"k",50931:"glu",51006:"bun",50882:"hco3"}
MAX_HR = 200  # 覆盖 48h+168h 检查点窗

def build_hourly(df, hr_col, val_col):
    """每(stay,var) → (MAX_HR,) 数组 + (MAX_HR,) 掩码"""
    out = {}
    for sid, gg in df.groupby("stay_id"):
        gg = gg.sort_values(hr_col, kind="mergesort")
        hrs = gg[hr_col].to_numpy(float)
        vals = gg[val_col].to_numpy(float)
        arr = np.zeros(MAX_HR, np.float32); msk = np.zeros(MAX_HR, np.float32)
        bins = np.clip(hrs.astype(int), 0, MAX_HR-1)
        valid = (hrs >= 0) & (hrs < MAX_HR)
        for b, v, ok in zip(bins, vals, valid):
            if ok: arr[b] = v; msk[b] = 1.0
        out[sid] = (arr, msk)
    return out

hourly = {}
for iid, nm in VIT_MAP.items():
    g = vit[vit.itemid==iid]
    if nm == "temp_c": g = g.assign(valuenum=(g.valuenum-32)/1.8)
    hourly[nm] = build_hourly(g, "hr", "valuenum")
for iid, nm in LAB_MAP.items():
    hourly[nm] = build_hourly(labs[labs.itemid==iid], "hr", "valuenum")
# UO: 每小时合计
uo_hourly = {}
for sid, gg in uo_rows.groupby("stay_id"):
    arr = np.zeros(MAX_HR, np.float32); msk = np.zeros(MAX_HR, np.float32)
    bins = np.clip(gg.hr_from_icu.to_numpy(float).astype(int), 0, MAX_HR-1)
    for b in bins: arr[b] = 1.0  # 标记有记录
    uo_hourly[sid] = (arr, msk)
# UO 实际量：逐行累加
for sid, gg in uo_rows.groupby("stay_id"):
    arr, msk = uo_hourly.get(sid, (np.zeros(MAX_HR,np.float32), np.zeros(MAX_HR,np.float32)))
    bins = np.clip(gg.hr_from_icu.to_numpy(float).astype(int), 0, MAX_HR-1)
    vals = gg.uo_signed.to_numpy(float)
    for b, v in zip(bins, vals): arr[b] = v; msk[b] = 1.0
    uo_hourly[sid] = (arr, msk)
hourly["uo"] = uo_hourly

# ============================ 张量化 ============================
print("构建张量 (75k × 48 × 24)...")
grid = F[["stay_id","t_hr","label","year"]].reset_index(drop=True)
tr_m = (grid.year <= 2017).values
tu_m = ((grid.year >= 2018) & (grid.year <= 2019)).values
iv_m = ((grid.year >= 2020) & (grid.year <= 2022)).values
y = grid.label.values.astype(np.float32)

def build_batch(mask):
    idx = np.where(mask)[0]
    Xs = np.zeros((len(idx), H, NV_TS*2), np.float32)
    sids = grid.stay_id.values[idx]; ts = grid.t_hr.values[idx]
    for k, (sid, t) in enumerate(zip(sids, ts)):
        h0 = int(t) - H; h1 = int(t)
        for j, var in enumerate(TS_VARS):
            tbl = hourly.get(var, {})
            if sid in tbl:
                arr, msk = tbl[sid]
                lo = max(0, h0); hi = min(MAX_HR, h1)
                if hi > lo:
                    Xs[k, lo-h0:hi-h0, j] = arr[lo:hi]
                    Xs[k, lo-h0:hi-h0, j+NV_TS] = msk[lo:hi]
    return Xs, y[idx]

Xtr, ytr = build_batch(tr_m)
Xtu, ytu = build_batch(tu_m)
Xiv, yiv = build_batch(iv_m)

mu = Xtr[:,:,:NV_TS][Xtr[:,:,:NV_TS]!=0].mean() if (Xtr[:,:,:NV_TS]!=0).any() else 0.0
sd = max(Xtr[:,:,:NV_TS][Xtr[:,:,:NV_TS]!=0].std() if (Xtr[:,:,:NV_TS]!=0).any() else 1.0, 1e-6)
for X in (Xtr, Xtu, Xiv): X[:,:,:NV_TS] = (X[:,:,:NV_TS]-mu)/sd
print(f"train={Xtr.shape}({int(ytr.sum())}+) select={Xtu.shape}({int(ytu.sum())}+) intval={Xiv.shape}({int(yiv.sum())}+)")

# ============================ 模型 ============================
class MaskedCNN(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.c1=nn.Conv1d(n_in,64,3,padding=1); self.b1=nn.BatchNorm1d(64)
        self.c2=nn.Conv1d(64,128,3,padding=1); self.b2=nn.BatchNorm1d(128)
        self.c3=nn.Conv1d(128,64,3,padding=1); self.b3=nn.BatchNorm1d(64)
        self.res=nn.Conv1d(n_in,64,1)
        self.pool=nn.AdaptiveMaxPool1d(1); self.drop=nn.Dropout(0.3); self.fc=nn.Linear(64,1)
    def forward(self,x):
        x=x.permute(0,2,1); r=self.res(x)
        h=torch.relu(self.b1(self.c1(x))); h=torch.relu(self.b2(self.c2(h)))
        h=torch.relu(self.b3(self.c3(h)))+r
        return self.fc(self.drop(self.pool(h).squeeze(-1))).squeeze(-1)

class LSTMAttention(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.lstm=nn.LSTM(n_in,64,num_layers=3,batch_first=True,dropout=0.2)
        self.attn=nn.Linear(64,1); self.drop=nn.Dropout(0.3); self.fc=nn.Linear(64,1)
    def forward(self,x):
        o,_=self.lstm(x); a=torch.softmax(self.attn(o),dim=1)
        return self.fc(self.drop((a*o).sum(dim=1))).squeeze(-1)

class MiniTransformer(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.embed=nn.Linear(n_in,64)
        self.pos=nn.Parameter(torch.randn(1,H,64)*0.02)
        self.cls=nn.Parameter(torch.randn(1,1,64)*0.02)
        layer=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=128,dropout=0.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2)
        self.drop=nn.Dropout(0.3); self.fc=nn.Linear(64,1)
    def forward(self,x):
        e=self.embed(x)+self.pos
        cls=self.cls.expand(x.size(0),-1,-1)
        h=self.encoder(torch.cat([cls,e],dim=1))
        return self.fc(self.drop(h[:,0])).squeeze(-1)

# ============================ 训练 ============================
def run(cls, tag):
    model=cls(NV_TS*2).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    pos_w=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],dtype=torch.float32)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pos_w)
    bs=32; best_val=-1; best_state=None; patience=0
    for ep in range(50):
        model.train()
        perm=np.random.permutation(len(ytr))
        for k in range(0,len(perm),bs):
            b=perm[k:k+bs]
            xb=torch.tensor(Xtr[b]).to(DEVICE); yb=torch.tensor(ytr[b]).to(DEVICE)
            opt.zero_grad(); lossf(model(xb),yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p_tu=torch.sigmoid(model(torch.tensor(Xtu))).numpy()
        val=roc_auc_score(ytu,p_tu)
        if val>best_val: best_val,patience=val,0
        else:
            patience+=1
            if patience>=10: break
    model.load_state_dict({k:v for k,v in model.state_dict().items()})  # 用最后 epoch
    model.eval()
    with torch.no_grad():
        p_iv=torch.sigmoid(model(torch.tensor(Xiv))).numpy()
    auc=roc_auc_score(yiv,p_iv)
    bs_ci=[roc_auc_score(yiv[np.random.default_rng(k).integers(0,len(yiv),len(yiv))],
            p_iv[np.random.default_rng(k).integers(0,len(yiv),len(yiv))])
           for k in range(1000) if len(set(yiv[np.random.default_rng(k).integers(0,len(yiv),len(yiv))]))>1]
    iv_df=pd.DataFrame({"sid":grid.stay_id.values[iv_m],"p":p_iv})
    pat_auc=roc_auc_score(grid[iv_m].groupby("stay_id").label.max().values,
                          iv_df.groupby("sid").p.max().values)
    r={"tag":tag,"select_auc":round(float(best_val),4),
       "intval_auroc":round(float(auc),4),
       "ci":[round(float(np.percentile(bs_ci,2.5)),4),round(float(np.percentile(bs_ci,97.5)),4)],
       "auprc":round(float(average_precision_score(yiv,p_iv)),4),
       "patient_level":round(float(pat_auc),4)}
    print(f"  {tag}: select={r['select_auc']} | intval AUROC={r['intval_auroc']} {r['ci']} | AUPRC={r['auprc']} | 患者级={r['patient_level']}")
    return r

results={}
for cls,tag in [(MaskedCNN,"masked_cnn"),(LSTMAttention,"lstm_attention"),(MiniTransformer,"mini_transformer")]:
    print(f"\n===== {tag} ====="); results[tag]=run(cls,tag)

results["gbm_baseline"]={"intval_auroc":0.7375,"ci":[0.7273,0.7474],"auprc":0.2497,"patient_level":0.7701}
out={"probe":"N7 ABI deep learning v2 (npjDM aligned)","date":"2026-09-21",
     "n_ts_vars":NV_TS,"vars":TS_VARS,**results}
with open(f"{R}/n7_deep_models.json","w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=1)
print("\n=== 横评 ===")
for tag,r in results.items():
    print(f"  {tag:20s}: AUROC {r['intval_auroc']} | AUPRC {r['auprc']} | 患者级 {r['patient_level']}")
