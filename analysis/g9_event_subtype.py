# -*- coding: utf-8 -*-
"""G9: 事件亚型分层验证（R1/R2 审稿 Must-Fix #1）
========================================
三分类：cr_only（仅 Cr 达标）/ uo_only（仅 UO 达标，Cr-only 终点永远看不见的事件）/ dual（双达标）
产出（M4 内验 + eICU 外验）：
  ① 患者级亚型计数 + 阳性检查点分解
  ② 同模型三终点判别：union（主终点）/ Cr-only 终点 / UO-only 终点 —— 同一网格重打标签
  ③ 亚型限定判别：该亚型事件检查点 vs 无事件患者检查点（干净负对照）
  ④ 0.20 工作点捕获/lead time 按亚型分层（g3 同款单触发定义）
  ⑤ 重建一致性门禁：M4 重建 min(cr_event, uo_onset) 与网格 onset_hr 不一致 >0.5% → stop()
数据真实性：全部数字来自本脚本实跑输出，落盘 g9_event_subtype.json
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import gate

gate(__file__)

from sklearn.metrics import roc_auc_score, average_precision_score

R = "E:/TBI subtype/09_tbi_aki/results"
W_START, W_END, FFILL_H, KDIGO_RATIO = 24.0, 168.0, 96.0, 1.5
_g2 = json.load(open(f"{R}/g2_recalibration_dca.json", encoding="utf-8"))["recal_layer_frozen"]
A, B = float(_g2["a_intercept"]), float(_g2["b_slope"])


def apply_layer(p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(A + B * z)))


# ---------- M4 事件轴重建（逐字复刻 n3 冻结规则） ----------
def rebuild_m4_event_axis():
    cr = pd.read_parquet(f"{R}/n2_m4_abi_cr_serial.parquet").dropna(subset=["valuenum"])
    uo_bk = pd.read_parquet(f"{R}/n3_m4_abi_uo_buckets.parquet")
    coh = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet")
    uo_onset = (uo_bk[(uo_bk.bucket_idx >= 4) & uo_bk.evaluable_strict & uo_bk.below_05_strict]
                .groupby("stay_id").t0_hr.min().rename("uo_hr"))

    import duckdb
    con = duckdb.connect()
    rrt = con.execute(f"""
    SELECT pe.stay_id, (epoch(pe.starttime)-epoch(c.intime))/3600.0 AS hr
    FROM read_csv_auto('E:/TBI subtype/data/mimic-iv-3.1/icu/procedureevents.csv.gz') pe
    JOIN read_parquet('{R}/n1_m4_abi_cohort.parquet') c USING(stay_id)
    WHERE pe.itemid IN (225441, 740477, 772042)
      AND pe.stay_id IN (SELECT stay_id FROM read_parquet('{R}/n1_m4_abi_cohort.parquet'))
    """).df()
    rrt_first = rrt.groupby("stay_id").hr.min()

    def ffill_event(g, base, cutoff):
        """返回 (cr_label, cr_event_hr)：n3 ffill_label 同逻辑，仅取事件口径"""
        end = cutoff if cutoff is not None else W_END
        if base != base or end <= W_START:
            return None, None
        hr = g.hr.to_numpy(float)
        val = g.valuenum.to_numpy(float)
        thresh = KDIGO_RATIO * base
        viw = (hr + FFILL_H > W_START) & (hr < end)
        if not viw.any():
            return False, np.nan  # no_eval（不会出现在网格）
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
            return True, np.nan  # evaluable=True, v2=0（ok0）
        inwin = [t for t in times if t > W_START]
        if not inwin:
            return None, None  # carryin（不在网格）
        return True, float(min(inwin))

    rows = []
    gmap = dict(tuple(cr.groupby("stay_id")))
    for sid in coh.stay_id:
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
        if lab is None or not lab:      # carryin / ok0 / no_eval：无 Cr 事件
            ehr = np.nan
        rows.append((sid, float(ehr) if ehr == ehr else np.nan))
    ax = pd.DataFrame(rows, columns=["stay_id", "cr_event_hr"])
    ax = ax.merge(uo_onset, on="stay_id", how="left")
    return ax


def subtype_of(cr_hr, uo_hr):
    has_cr, has_uo = cr_hr == cr_hr, uo_hr == uo_hr
    if has_cr and has_uo:
        return "dual"
    if has_cr:
        return "cr_only"
    if has_uo:
        return "uo_only"
    return "none"


def analyse(preds, grid, ax, tag, thr_report=(0.10, 0.20)):
    """ax: stay_id, cr_event_hr, uo_hr；grid: stay_id,t_hr,label,onset_hr"""
    d = preds.merge(grid, on=["stay_id", "t_hr"], how="inner")
    d = d.merge(ax, on="stay_id", how="left")
    d["subtype"] = [subtype_of(a, b) for a, b in zip(d.cr_event_hr, d.uo_hr)]
    d["p_cal"] = apply_layer(d.p.values)

    # ---- ① 计数 ----
    st = d.groupby("stay_id").agg(
        subtype=("subtype", "first"), onset=("onset_hr", "first"),
        cr=("cr_event_hr", "first"), uo=("uo_hr", "first"))
    n_pat = {s: int((st.subtype == s).sum()) for s in ["cr_only", "uo_only", "dual", "none"]}
    # 检查点级三终点标签
    d["y_union"] = d.label.astype(int)
    d["y_cr"] = ((d.cr_event_hr == d.cr_event_hr) &
                 (d.cr_event_hr - d.t_hr > 0) & (d.cr_event_hr - d.t_hr <= 48)).astype(int)
    d["y_uo"] = ((d.uo_hr == d.uo_hr) &
                 (d.uo_hr - d.t_hr > 0) & (d.uo_hr - d.t_hr <= 48)).astype(int)
    pos_ckpt = {k: int(d[f"y_{k}"].sum()) for k in ["union", "cr", "uo"]}

    # ---- ② 同模型三终点判别（全网格） ----
    tri = {}
    for k in ["union", "cr", "uo"]:
        y = d[f"y_{k}"].values
        tri[k] = {"auroc": round(float(roc_auc_score(y, d.p.values)), 4),
                  "auprc": round(float(average_precision_score(y, d.p.values)), 4),
                  "pos": int(y.sum()), "pos_rate": round(float(y.mean()), 4)}

    # ---- ③ 亚型限定判别（该亚型阳性检查点 vs 无事件患者全部检查点） ----
    sub_disc = {}
    for s in ["cr_only", "uo_only", "dual"]:
        pos = d[(d.subtype == s) & (d.onset_hr == d.onset_hr) &
                (d.onset_hr - d.t_hr > 0) & (d.onset_hr - d.t_hr <= 48)]
        neg = d[d.subtype == "none"]
        yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        pp = np.r_[pos.p.values, neg.p.values]
        if len(pos) and yy.sum() < len(yy):
            sub_disc[s] = {"n_pos_ckpt": int(len(pos)), "n_neg_ckpt": int(len(neg)),
                           "auroc": round(float(roc_auc_score(yy, pp)), 4),
                           "auprc": round(float(average_precision_score(yy, pp)), 4)}
        else:
            sub_disc[s] = {"n_pos_ckpt": int(len(pos)), "n_neg_ckpt": int(len(neg)),
                           "auroc": None, "auprc": None}

    # ---- ④ 0.20 捕获/lead time 按亚型（g3 同款：onset 前任一触发即捕获） ----
    cap = {}
    G = {sid: (g.t_hr.to_numpy(float), g.p_cal.to_numpy(float))
         for sid, g in d.groupby("stay_id", sort=False)}
    for thr in thr_report:
        rows_t = []
        for s in ["cr_only", "uo_only", "dual"]:
            ev = st[(st.subtype == s) & (st.onset == st.onset)]
            captured, leads = 0, []
            for sid, onset in zip(ev.index, ev.onset):
                ts, ps = G[sid]
                pre = (ts < onset) & (ps >= thr)
                if pre.any():
                    captured += 1
                    leads.append(float(onset - ts[pre][0]))
            rows_t.append({
                "subtype": s, "n_events": int(len(ev)), "captured": int(captured),
                "capture_rate": round(captured / max(len(ev), 1), 3),
                "lead_median_h": round(float(np.median(leads)), 1) if leads else None,
                "lead_iqr_h": [round(float(np.percentile(leads, 25)), 1),
                               round(float(np.percentile(leads, 75)), 1)] if leads else None})
        cap[f"thr{thr:.2f}"] = rows_t

    return {"tag": tag, "patients": n_pat, "positive_checkpoints": pos_ckpt,
            "endpoint_discrimination": tri, "subtype_restricted": sub_disc,
            "capture_by_subtype": cap}, d


# ================= M4 内验 =================
print("=== M4 内验：重建事件轴 ===")
ax_m4 = rebuild_m4_event_axis()
grid_m4 = pd.read_parquet(f"{R}/n3_m4_abi_grid.parquet")
# 一致性门禁：网格 onset vs min(重建 cr_event, uo_onset)
chk = grid_m4.groupby("stay_id").onset_hr.first().rename("onset_grid").to_frame()
chk = chk.join(ax_m4.set_index("stay_id"))
chk["onset_recon"] = chk[["cr_event_hr", "uo_hr"]].min(axis=1)
mism = ((chk.onset_grid.isna() != chk.onset_recon.isna()) |
        ((chk.onset_grid.isna() == False) &
         (chk.onset_grid - chk.onset_recon).abs().gt(1e-9))).sum()
rate = float(mism) / len(chk)
print(f"重建一致性: {mism}/{len(chk)} 不一致 ({rate:.3%})")
if rate > 0.005:
    bad = chk[(chk.onset_grid.isna() != chk.onset_recon.isna()) |
              ((chk.onset_grid.isna() == False) & (chk.onset_grid - chk.onset_recon).abs().gt(1e-9))]
    print(bad.head(10))
    raise SystemExit("GATE FAIL: M4 事件轴重建与网格不一致 >0.5%")

preds_m4 = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet").rename(columns={"y": "label_y"})
preds_m4 = preds_m4.drop(columns=["label_y"])
res_m4, _ = analyse(preds_m4, grid_m4, ax_m4, "M4_internal_validation")
print(json.dumps({k: res_m4[k] for k in ["patients", "positive_checkpoints"]}, indent=1))

# ================= eICU 外验 =================
print("=== eICU 外验 ===")
ax_e = pd.read_parquet(f"{R}/n8_3_eicu_abi_event_axis.parquet")
ax_e = ax_e[["stay_id", "event_hr_final", "uo_onset_hr"]].rename(
    columns={"event_hr_final": "cr_event_hr", "uo_onset_hr": "uo_hr"})
grid_e = pd.read_parquet(f"{R}/n8_3_eicu_abi_grid.parquet")
preds_e = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")[["stay_id", "t_hr", "p"]]
res_e, _ = analyse(preds_e, grid_e, ax_e, "eICU_external_validation")
print(json.dumps({k: res_e[k] for k in ["patients", "positive_checkpoints"]}, indent=1))

out = {"probe": "G9 event-subtype stratified validation (reviewer Must-Fix 1)",
       "date": "2026-09-23",
       "reconstruction_gate": {"m4_mismatch": int(mism), "n_stays": int(len(chk)),
                                "mismatch_rate": round(rate, 5)},
       "m4_internal_validation": res_m4,
       "eicu_external_validation": res_e}
with open(f"{R}/g9_event_subtype.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\n=== G9 完成 → results/g9_event_subtype.json ===")
print(json.dumps(out["m4_internal_validation"]["capture_by_subtype"]["thr0.20"], indent=1))
print(json.dumps(out["eicu_external_validation"]["capture_by_subtype"]["thr0.20"], indent=1))
