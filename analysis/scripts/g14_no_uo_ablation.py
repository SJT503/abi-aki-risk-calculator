# -*- coding: utf-8 -*-
"""G14: 去尿量消融（R5 循环性攻击的正面实验回答）
==========================================================
问题：UO 特征预测含 UO 定义的终点，是否存在定义循环性？
设计：从冻结 291 特征中剔除全部 17 个 UO/流体特征 → 274 特征重训
     （逐字镜像 n6 协议：同 3 HP / select 集选优 / class_weight balanced / seed 42）
评估（M4 内验 + eICU 外验）：
  ① 三终点判别：union / cr-only / uo-only（亚型终点用 g9 同款轴重建）
  ② 亚型限定判别：uo_only 事件 vs 无事件患者（循环性核心检验）
  ③ uo_only 捕获率 @0.20（消融模型自配 select-fit 重校准层，同 g2 协议）
  ④ 配对 ΔAUROC（full − no-UO），检查点级 bootstrap 1000 次，内外验
门禁：
  - 消融特征数必须 = 291 − 17 = 274，否则 stop
  - 冻结模型预测复现门禁：full preds AUROC 必须复现 0.7369 / 0.7094
  - M4 事件轴重建一致性 >0.5% 不一致 → stop（g9 同款）
数据真实性：全部数字来自本脚本实跑输出，落盘 g14_no_uo_ablation.json
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import statsmodels.api as sm

SEED = 42
R = "E:/TBI subtype/09_tbi_aki/results"
W_START, W_END, FFILL_H, KDIGO_RATIO = 24.0, 168.0, 96.0, 1.5

UO17 = ["intake_ml_24h", "net_balance_ml_kg", "uo_cv", "uo_delta", "uo_h_last",
        "uo_last", "uo_locf", "uo_locf_age", "uo_max", "uo_min", "uo_ml_kg_h_12h",
        "uo_ml_kg_h_24h", "uo_ml_kg_h_6h", "uo_n", "uo_sd", "uo_slope", "uo_tot48"]

# ================= 数据装载 + 年份还原（镜像 n6） =================
F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
import duckdb
con = duckdb.connect()
adm = con.execute(f"""
SELECT c.stay_id, c.subject_id, a.admittime, p.anchor_year
FROM read_parquet('{R}/n1_m4_abi_cohort.parquet') c
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/admissions.csv.gz') a ON c.hadm_id = a.hadm_id
JOIN read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz') p ON c.subject_id = p.subject_id
""").df()
pat = pd.read_csv("E:/TBI subtype/data/mimic-iv-3.1/hosp/patients.csv.gz",
                  usecols=["subject_id", "anchor_year_group"])
_g = pat["anchor_year_group"].astype(str).str.extract(r"(\d{4})\D+(\d{4})")
pat["group_mid"] = (_g[0].astype(int) + _g[1].astype(int)) / 2.0
coh = adm.merge(pat[["subject_id", "group_mid"]], on="subject_id", how="left")
coh["year"] = (coh.group_mid + (coh.admittime.dt.year - coh.anchor_year)).round().astype(int)
F = F.merge(coh[["stay_id", "year"]], on="stay_id", how="left")

frozen = joblib.load(f"{R}/n6_abi_gbm_frozen.joblib")
feats_full = frozen["feats"]
ablation_feats = [f for f in feats_full if f not in UO17]
print(f"[gate] 冻结特征 {len(feats_full)} − UO17 中在集 {sum(1 for x in UO17 if x in feats_full)} = 消融特征 {len(ablation_feats)}")
if len(ablation_feats) != len(feats_full) - 17:
    raise SystemExit(f"GATE FAIL: 消融特征数 {len(ablation_feats)} ≠ 291-17=274")

tr = F[F.year <= 2017]
tu = F[(F.year >= 2018) & (F.year <= 2019)]
iv = F[(F.year >= 2020) & (F.year <= 2022)]
ytr, ytu, yiv = tr.label.values, tu.label.values, iv.label.values
print(f"train={len(tr)}({int(ytr.sum())}+) select={len(tu)}({int(ytu.sum())}+) intval={len(iv)}({int(yiv.sum())}+)")

# ================= 消融模型训练（镜像 n6：3 HP / select 选优） =================
best_auc, best_m = -1, None
for ne, lr, nl in [(500, .03, 63), (800, .02, 127), (600, .05, 31)]:
    m = lgb.LGBMClassifier(n_estimators=ne, learning_rate=lr, num_leaves=nl,
                           class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(tr[ablation_feats], ytr)
    a = roc_auc_score(ytu, m.predict_proba(tu[ablation_feats])[:, 1])
    if a > best_auc:
        best_auc, best_m = a, m
    print(f"  hp=({ne},{lr},{nl}): select AUROC={a:.4f}")

p_iv_ab = best_m.predict_proba(iv[ablation_feats])[:, 1]

E = pd.read_parquet(f"{R}/n8_4_eicu_abi_features.parquet")
p_ex_ab = best_m.predict_proba(E.reindex(columns=ablation_feats))[:, 1]

# ---- 冻结模型预测复现门禁（full preds 必须复现已发表聚合值） ----
fiv = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet")
fex = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")
auc_fiv = round(float(roc_auc_score(fiv.y, fiv.p)), 4)
auc_fex = round(float(roc_auc_score(fex.y, fex.p)), 4)
print(f"[gate] full 复现: intval={auc_fiv} (期望0.7369) | eICU={auc_fex} (期望0.7094)")
if (auc_fiv != 0.7369) or (auc_fex != 0.7094):
    raise SystemExit("GATE FAIL: 冻结模型预测未能复现已发表 AUROC")

# ================= 消融模型自配 select-fit 重校准层（镜像 g2 协议） =================
z_tu = np.log(np.clip(best_m.predict_proba(tu[ablation_feats])[:, 1], 1e-6, 1 - 1e-6)
              / (1 - np.clip(best_m.predict_proba(tu[ablation_feats])[:, 1], 1e-6, 1 - 1e-6)))
cal = sm.Logit(ytu, sm.add_constant(z_tu)).fit(disp=0)
A_ab, B_ab = float(cal.params[0]), float(cal.params[1])
print(f"[recal] 消融层: a={A_ab:.4f} b={B_ab:.4f} (full 层参考: a=-0.5414 b=0.3569)")


def apply_layer_ab(p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(A_ab + B_ab * z)))


# ================= M4 事件轴重建（g9 逐字复刻） =================
def rebuild_m4_event_axis():
    cr = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet").dropna(subset=["valuenum"])
    uo_bk = pd.read_parquet(f"{R}/n3_m4_abi_uo_buckets.parquet")
    cohq = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet")
    uo_onset = (uo_bk[(uo_bk.bucket_idx >= 4) & uo_bk.evaluable_strict & uo_bk.below_05_strict]
                .groupby("stay_id").t0_hr.min().rename("uo_hr"))
    rrt = con.execute(f"""
    SELECT pe.stay_id, (epoch(pe.starttime)-epoch(c.intime))/3600.0 AS hr
    FROM read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/icu/procedureevents.csv.gz') pe
    JOIN read_parquet('{R}/n1_m4_abi_cohort.parquet') c USING(stay_id)
    WHERE pe.itemid IN (225441, 740477, 772042)
      AND pe.stay_id IN (SELECT stay_id FROM read_parquet('{R}/n1_m4_abi_cohort.parquet'))
    """).df()
    rrt_first = rrt.groupby("stay_id").hr.min()

    def ffill_event(g, base, cutoff):
        end = cutoff if cutoff is not None else W_END
        if base != base or end <= W_START:
            return None, None
        hr = g.hr.to_numpy(float)
        val = g.valuenum.to_numpy(float)
        thresh = KDIGO_RATIO * base
        viw = (hr + FFILL_H > W_START) & (hr < end)
        if not viw.any():
            return False, np.nan
        t_eff = np.maximum(hr, W_START)
        times = [float(t_eff[k]) for k in np.where(viw & (val >= thresh))[0]]
        j0 = 0
        for i in range(len(hr)):
            while hr[i] - hr[j0] > 48:
                j0 += 1
            if val[i] - val[j0:i + 1].min() >= 0.3:
                t = float(max(hr[i], W_START))
                if W_START < t < end:
                    times.append(t)
        if not times:
            return True, np.nan
        inwin = [t for t in times if t > W_START]
        if not inwin:
            return None, None
        return True, float(min(inwin))

    rows = []
    gmap = dict(tuple(cr.groupby("stay_id")))
    for sid in cohq.stay_id:
        g = gmap.get(sid)
        if g is None:
            continue
        g = g.sort_values(["hr", "valuenum"], kind="mergesort")
        pre = g[g.hr < 0]
        win24 = g[(g.hr >= 0) & (g.hr <= 24)]
        if len(win24):
            base = float(win24.valuenum.min())
        elif len(pre):
            base = float(pre.valuenum.min())
        else:
            base = np.nan
        if base != base:
            continue
        rrt_hr = float(rrt_first[sid]) if sid in rrt_first.index else np.nan
        cutoff = rrt_hr if (rrt_hr == rrt_hr and rrt_hr <= W_END) else None
        lab, ehr = ffill_event(g, base, cutoff)
        if lab is None or not lab:
            ehr = np.nan
        rows.append((sid, float(ehr) if ehr == ehr else np.nan))
    ax = pd.DataFrame(rows, columns=["stay_id", "cr_event_hr"])
    return ax.merge(uo_onset, on="stay_id", how="left")


print("=== M4 事件轴重建 ===")
ax_m4 = rebuild_m4_event_axis()
grid_m4 = pd.read_parquet(f"{R}/n3_m4_abi_grid.parquet")
chk = grid_m4.groupby("stay_id").onset_hr.first().rename("onset_grid").to_frame()
chk = chk.join(ax_m4.set_index("stay_id"))
chk["onset_recon"] = chk[["cr_event_hr", "uo_hr"]].min(axis=1)
mism = ((chk.onset_grid.isna() != chk.onset_recon.isna()) |
        ((chk.onset_grid.isna() == False) & (chk.onset_grid - chk.onset_recon).abs().gt(1e-9))).sum()
rate = float(mism) / len(chk)
print(f"重建一致性: {mism}/{len(chk)} ({rate:.3%})")
if rate > 0.005:
    raise SystemExit("GATE FAIL: M4 事件轴重建不一致 >0.5%")


def subtype_of(cr_hr, uo_hr):
    has_cr, has_uo = cr_hr == cr_hr, uo_hr == uo_hr
    if has_cr and has_uo:
        return "dual"
    if has_cr:
        return "cr_only"
    if has_uo:
        return "uo_only"
    return "none"


def analyse(stay, t, p_full, p_ab, grid, ax, tag):
    d = pd.DataFrame({"stay_id": stay, "t_hr": t, "p_full": p_full, "p_ab": p_ab})
    d = d.merge(grid, on=["stay_id", "t_hr"], how="inner")
    d = d.merge(ax, on="stay_id", how="left")
    d["subtype"] = [subtype_of(a, b) for a, b in zip(d.cr_event_hr, d.uo_hr)]
    d["p_ab_cal"] = apply_layer_ab(d.p_ab.values)
    d["y_union"] = d.label.astype(int)
    d["y_cr"] = ((d.cr_event_hr == d.cr_event_hr) & (d.cr_event_hr - d.t_hr > 0)
                 & (d.cr_event_hr - d.t_hr <= 48)).astype(int)
    d["y_uo"] = ((d.uo_hr == d.uo_hr) & (d.uo_hr - d.t_hr > 0)
                 & (d.uo_hr - d.t_hr <= 48)).astype(int)

    tri = {}
    for k in ["union", "cr", "uo"]:
        y = d[f"y_{k}"].values
        tri[k] = {"full": round(float(roc_auc_score(y, d.p_full.values)), 4),
                  "no_uo": round(float(roc_auc_score(y, d.p_ab.values)), 4),
                  "pos": int(y.sum())}

    sub_disc = {}
    for s in ["cr_only", "uo_only", "dual"]:
        pos = d[(d.subtype == s) & (d.onset_hr == d.onset_hr) &
                (d.onset_hr - d.t_hr > 0) & (d.onset_hr - d.t_hr <= 48)]
        neg = d[d.subtype == "none"]
        yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        sub_disc[s] = {"n_pos_ckpt": int(len(pos)),
                       "full": round(float(roc_auc_score(yy, np.r_[pos.p_full.values, neg.p_full.values])), 4),
                       "no_uo": round(float(roc_auc_score(yy, np.r_[pos.p_ab.values, neg.p_ab.values])), 4)}

    # uo_only 捕获 @0.20（消融模型重校准概率，g3 同款单触发定义）
    st = d.groupby("stay_id", as_index=False).agg(subtype=("subtype", "first"), onset=("onset_hr", "first"))
    G = {sid: (g.t_hr.to_numpy(float), g.p_ab_cal.to_numpy(float)) for sid, g in d.groupby("stay_id", sort=False)}
    cap = {}
    for s in ["cr_only", "uo_only", "dual"]:
        ev = st[(st.subtype == s) & (st.onset == st.onset)]
        captured, leads = 0, []
        for sid, onset in zip(ev.stay_id, ev.onset):
            ts, ps = G[sid]
            pre = (ts < onset) & (ps >= 0.20)
            if pre.any():
                captured += 1
                leads.append(float(onset - ts[pre][0]))
        cap[s] = {"n_events": int(len(ev)), "captured": int(captured),
                  "capture_rate": round(captured / max(len(ev), 1), 3),
                  "lead_median_h": round(float(np.median(leads)), 1) if leads else None}

    # 配对 ΔAUROC（full − no_uo）@union 终点，检查点级 bootstrap 1000
    y = d.y_union.values
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            deltas.append(roc_auc_score(y[idx], d.p_full.values[idx]) - roc_auc_score(y[idx], d.p_ab.values[idx]))
    pd_delta = {"median": round(float(np.median(deltas)), 4),
                "ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                         round(float(np.percentile(deltas, 97.5)), 4)]}

    return {"tag": tag, "n_ckpt": int(len(d)),
            "endpoint_discrimination": tri, "subtype_restricted": sub_disc,
            "capture02_no_uo": cap, "paired_delta_full_minus_nouo": pd_delta}, d


# ---- M4 内验 ----
res_m4, d_m4 = analyse(iv.stay_id.values, iv.t_hr.values, fiv.p.values, p_iv_ab,
                       grid_m4, ax_m4, "M4_internal_validation")

# ---- eICU 外验 ----
ax_e = pd.read_parquet(f"{R}/n8_3_eicu_abi_event_axis.parquet")[["stay_id", "event_hr_final", "uo_onset_hr"]].rename(
    columns={"event_hr_final": "cr_event_hr", "uo_onset_hr": "uo_hr"})
grid_e = pd.read_parquet(f"{R}/n8_3_eicu_abi_grid.parquet")
res_e, d_e = analyse(E.stay_id.values, E.t_hr.values, fex.p.values, p_ex_ab,
                     grid_e, ax_e, "eICU_external_validation")

out = {"probe": "G14 no-UO ablation (circularity answer to R5)",
       "date": "2026-09-23",
       "design": {"n_features_full": len(feats_full), "uo_features_removed": 17,
                  "n_features_ablated": len(ablation_feats),
                  "protocol": "n6 verbatim (3 HP, select-fit, balanced, seed 42)",
                  "recal_layer_ablated": {"a": round(A_ab, 4), "b": round(B_ab, 4)}},
       "reproduction_gate": {"full_intval": auc_fiv, "full_eicu": auc_fex,
                             "m4_axis_mismatch_rate": round(rate, 5)},
       "m4_internal_validation": res_m4,
       "eicu_external_validation": res_e}
with open(f"{R}/g14_no_uo_ablation.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

pd.concat([
    pd.DataFrame({"stay_id": d_m4.stay_id, "t_hr": d_m4.t_hr, "y": d_m4.y_union, "p": d_m4.p_ab, "set": "M4_intval"}),
    pd.DataFrame({"stay_id": d_e.stay_id, "t_hr": d_e.t_hr, "y": d_e.y_union, "p": d_e.p_ab, "set": "eICU_external"}),
]).to_parquet(f"{R}/g14_nouo_preds.parquet", index=False)

print("\n=== G14 完成 → results/g14_no_uo_ablation.json ===")
print(json.dumps({"M4": res_m4["endpoint_discrimination"], "eICU": res_e["endpoint_discrimination"],
                  "M4_uo_restricted": res_m4["subtype_restricted"]["uo_only"],
                  "eICU_uo_restricted": res_e["subtype_restricted"]["uo_only"],
                  "M4_delta": res_m4["paired_delta_full_minus_nouo"],
                  "eICU_delta": res_e["paired_delta_full_minus_nouo"]}, indent=1))
