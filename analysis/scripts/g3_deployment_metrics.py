# -*- coding: utf-8 -*-
"""G3: P0-c 早期预警四视角部署指标（skill Step 10 四视角体系；npj DM AKI 先例同病种对垒）
概率源 = 冻结模型 × select-fit 重校准层（g2 部署链，判别不变）
① 事件端 lead time（事件患者首触发→onset，小时）
② 系统端 假警报率（非事件患者的警报 episode / 100 无事件病人日）
③ 行动端 NNE（总警报 episode / 捕获事件数 ≈ 1/PPV）——单次触发 vs 双次连续触发两口径
④ 时间端 Mann-Kendall τ（按 t 分桶 AUROC：数据积累是否改善性能）
+ 12h 预警窗敏感性/特异性
事件/无事件口径：网格内 onset 存在=事件患者；无事件=网格走完无 onset
输出：results/g3_deployment_metrics.json
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

from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

R = "E:/TBI subtype/09_tbi_aki/results"
THR = [0.10, 0.20, 0.30]
_g2 = json.load(open(f"{R}/g2_recalibration_dca.json", encoding="utf-8"))["recal_layer_frozen"]
A, B = float(_g2["a_intercept"]), float(_g2["b_slope"])  # 审计修正：从 g2 json 读层系数（防重冻结后漂移）


def apply_layer(p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(A + B * z)))


def episodes(trigger_flags):
    """连续触发 run 数（episode）：[1,1,0,1]→2；[0,1,0,1]→2（核验员修正 off-by-one：首元素为触发才计基数）"""
    t = np.asarray(trigger_flags, dtype=int)
    if t.sum() == 0:
        return 0
    return int((t[0] == 1) + np.sum((t[1:] == 1) & (t[:-1] == 0)))


def four_view(pred, grid, tag):
    d = pred.merge(grid[["stay_id", "t_hr", "label", "onset_hr"]], on=["stay_id", "t_hr"], how="inner")
    d = d.sort_values(["stay_id", "t_hr"], kind="mergesort")
    d["p_cal"] = apply_layer(d.p.values)
    onset_map = d.groupby("stay_id").onset_hr.first().to_dict()
    event_stays = {s for s, o in onset_map.items() if o == o}
    # 单次分组字典化（性能：避免逐 stay 布尔索引扫全表）
    G = {sid: (g.t_hr.to_numpy(float), g.p_cal.to_numpy(float), g.label.to_numpy(int))
         for sid, g in d.groupby("stay_id", sort=False)}
    res = {"n_event_patients": len(event_stays),
           "n_nonevent_patients": int(len(G) - len(event_stays))}
    for thr in THR:
        leads, captured, epi_total_false, epi_total_all, pdays = [], 0, 0, 0, 0.0
        leads2, captured2, epi2_false, epi2_all = [], 0, 0, 0
        for sid, (ts, ps, ys) in G.items():
            trig = (ps >= thr).astype(int)
            run2 = np.convolve(trig, [1, 1], "valid") == 2
            epi2_all += episodes(run2.astype(int))
            epi_total_all += episodes(trig)
            if sid in event_stays:
                onset = onset_map[sid]
                pre = trig[ts < onset]
                hits = np.where(pre == 1)[0]
                if len(hits):
                    captured += 1
                    leads.append(float(onset - ts[ts < onset][hits[0]]))
                pre2 = run2[ts[:-1] < onset]
                h2 = np.where(pre2)[0]
                if len(h2):
                    captured2 += 1
                    leads2.append(float(onset - ts[:-1][ts[:-1] < onset][h2[0]]))
            else:
                epi_total_false += episodes(trig)
                epi2_false += episodes(run2.astype(int))
                pdays += (ts.max() + 6 - ts.min()) / 24.0
        n_le12 = sum(1 for L in leads if L >= 12)
        res[f"thr{thr:.2f}"] = {
            "lead_time_median_h": round(float(np.median(leads)), 1) if leads else None,
            "lead_time_iqr_h": [round(float(np.percentile(leads, 25)), 1), round(float(np.percentile(leads, 75)), 1)] if leads else None,
            "capture_rate": round(captured / max(len(event_stays), 1), 3),
            "false_alarms_per_100ptdays": round(100 * epi_total_false / max(pdays, 1e-9), 1),
            "NNE_single": round(epi_total_all / max(captured, 1), 1),
            "NNE_double": round(epi2_all / max(captured2, 1), 1),
            "capture_double": round(captured2 / max(len(event_stays), 1), 3),
            "sens_flagged_ge12h_before_onset": round(n_le12 / max(len(event_stays), 1), 3)}
        r = res[f"thr{thr:.2f}"]
        print(f"  [{tag}] thr={thr:.2f}: lead={r['lead_time_median_h']}h capture={r['capture_rate']} "
              f"NNE={r['NNE_single']}/{r['NNE_double']} FA/100d={r['false_alarms_per_100ptdays']}")
    # ④ Mann-Kendall：按 t 分桶（24h 一桶）AUROC 趋势
    d["bucket"] = ((d.t_hr - 24) // 24).astype(int)
    ba = {}
    for b, g in d.groupby("bucket"):
        if g.label.nunique() > 1:
            ba[int((b + 1) * 24)] = round(float(roc_auc_score(g.label, g.p_cal)), 4)
    hours = sorted(ba)
    tau, p = kendalltau(hours, [ba[h] for h in hours])
    res["auroc_by_checkpoint_hour"] = ba
    res["mann_kendall"] = {"tau": round(float(tau), 3), "p": round(float(p), 4),
                           "reading": "tau>0 显著=数据积累改善（faithful）；tau≈0=平稳可用"}
    return res


out = {"probe": "G3 four-view deployment metrics (P0-c)", "date": "2026-09-21",
       "prob_chain": "frozen GBM × select-fit recal layer (g2)"}
iv_pred = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet")
iv_grid = pd.read_parquet(f"{R}/n3_m4_abi_grid.parquet")
ex_pred = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")
ex_grid = pd.read_parquet(f"{R}/n8_3_eicu_abi_grid.parquet")
print("== M4 内验 ==")
out["M4_intval"] = four_view(iv_pred, iv_grid, "M4")
print("== eICU 外验 ==")
out["eICU_external"] = four_view(ex_pred, ex_grid, "eICU")
with open(f"{R}/g3_deployment_metrics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("SAVED g3_deployment_metrics.json")
