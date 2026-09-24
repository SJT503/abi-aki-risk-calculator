# -*- coding: utf-8 -*-
"""G4: P1 可解释性+公平性（skill Step 9 / 11.1-11.2）
① SHAP 全局重要性（TreeExplainer，intval 8,000 行 rng(42) 抽样——全量不可行；DUA 红线：只输出聚合统计，不留个体矩阵）
② top-3 动态特征依赖曲线数据（十分位 bin → mean SHAP；手稿画图用数据）
③ SHAP 交互 top-5 对（2,000 行抽样，聚合；skill 11.2）
④ 公平性亚组 AUROC（age<65/65-79/≥80 × sex；M4 内验 + eICU 外验双库）
输出：results/g4_shap_fairness.json
"""
import json
import os
import sys
import warnings

import duckdb
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

import shap
from sklearn.metrics import roc_auc_score

R = "E:/TBI subtype/09_tbi_aki/results"
frozen = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")
feats, model = frozen["feats"], frozen["model"]

# ---- M4 内验集（同 g1 口径） ----
F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
con = duckdb.connect()
adm = con.execute(f"""
SELECT c.stay_id, c.subject_id, a.admittime, p.anchor_year
FROM read_parquet('{R}/n1_m4_abi_cohort.parquet') c
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/admissions.csv.gz') a ON c.hadm_id = a.hadm_id
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz') p ON c.subject_id = p.subject_id
""").df()
pat2 = pd.read_csv("E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz", usecols=["subject_id", "anchor_year_group"])
_g = pat2["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
pat2["group_mid"] = (_g[0].astype(int) + _g[1].astype(int)) / 2.0
coh = adm.merge(pat2[["subject_id", "group_mid"]], on="subject_id", how="left")
coh["year"] = (coh.group_mid + (coh.admittime.dt.year - coh.anchor_year)).round().astype(int)
F = F.merge(coh[["stay_id", "year"]], on="stay_id", how="left")
iv = F[(F.year >= 2020) & (F.year <= 2022)].reset_index(drop=True)
Xiv = iv[feats]
print(f"intval: {len(iv)} 行 / {iv.stay_id.nunique()} 患者")

# ---- ① SHAP 全局（8,000 行随机抽样——全量 28,885×291×800 树在本机 >20min 不可行；抽样 8k 足稳 mean|SHAP|） ----
rng = np.random.default_rng(42)
_sidx = rng.integers(0, len(iv), 8000)
Xs = Xiv.iloc[_sidx]
expl = shap.TreeExplainer(model)
sv = expl.shap_values(Xs)
if isinstance(sv, list):
    sv = sv[1]
imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
top20 = {k: round(float(v), 5) for k, v in imp.head(20).items()}
print("SHAP top10:", list(imp.head(10).index))

# ---- ② top-3 动态特征依赖曲线（排除 age/male/charlson 静态项） ----
DYNAMIC_EXCL = {"age", "male", "charlson", "cr_base", "t_hr"}
top_dyn = [f for f in imp.index if f not in DYNAMIC_EXCL][:3]
dep = {}
_sdf = iv.iloc[_sidx].reset_index(drop=True)  # 与 sv 同为 8,000 行抽样
for f in top_dyn:
    v = _sdf[f].values
    m = np.isfinite(v)
    q = pd.qcut(v[m], 10, duplicates="drop")
    df_ = pd.DataFrame({"v": v[m], "s": sv[:, feats.index(f)][m], "b": q})
    g = df_.groupby("b", observed=True).agg(v_mid=("v", "median"), s_mean=("s", "mean"), n=("v", "size"))
    dep[f] = {"bins": [round(float(x), 3) for x in g.v_mid],
              "mean_shap": [round(float(x), 4) for x in g.s_mean],
              "n_per_bin": [int(x) for x in g.n]}
print("dependence:", {f: dep[f]["mean_shap"] for f in top_dyn})

# ---- ③ SHAP 交互：不做（计算不可行，如实记录） ----
# 291 特征的 interaction TreeSHAP 复杂度 ~O(F²×trees) 每行，本机实测 2,000 行 >20min 未完成（D35 后审计轮）
# skill 11.2 交互分析为可选项；手稿 Methods 声明 not performed（computational infeasibility at F=291）
interaction_note = {"performed": False,
                    "reason": "interaction TreeSHAP at 291 features computationally infeasible on this hardware (>20min/2000 rows, aborted)"}
print("interaction: skipped (infeasible, documented)")

# ---- ④ 公平性亚组（M4 内验 + eICU 外验） ----
def fairness(df, p, tag):
    out = {}
    gs = [("age<65", df.age < 65), ("age 65-79", (df.age >= 65) & (df.age < 80)),
          ("age>=80", df.age >= 80), ("male", df.male == 1), ("female", df.male == 0)]
    for nm, m in gs:
        mm = m.values
        if mm.sum() > 100 and len(np.unique(df.label.values[mm])) > 1:
            out[nm] = {"n_ckpt": int(mm.sum()),
                       "auroc": round(float(roc_auc_score(df.label.values[mm], np.asarray(p)[mm])), 4)}
    return out

fair_m4 = fairness(iv, model.predict_proba(Xiv)[:, 1], "M4")
ex_pred = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")
ex_feat = pd.read_parquet(f"{R}/n8_4_eicu_abi_features.parquet", columns=["stay_id", "t_hr", "age", "male"])
exd = ex_pred.merge(ex_feat, on=["stay_id", "t_hr"], how="left")
exd = exd.rename(columns={"y": "label"})
fair_ex = fairness(exd, exd.p.values, "eICU")
print("fairness M4:", fair_m4)
print("fairness eICU:", fair_ex)

out = {"probe": "G4 SHAP + fairness (P1, Step 9)", "date": "2026-09-22",
       "shap": {"n_rows_sampled": 8000, "n_rows_total": len(iv), "top20_mean_abs": top20,
                "note": "聚合统计；个体矩阵不入库（DUA）；8k 随机抽样（全量不可行）"},
       "dependence_top3": dep,
       "interaction": interaction_note,
       "fairness": {"M4_intval": fair_m4, "eICU_external": fair_ex,
                    "rule": "亚组 AUROC 差距<0.05 或 Discussion 解释"}}
with open(f"{R}/g4_shap_fairness.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("SAVED g4_shap_fairness.json")
