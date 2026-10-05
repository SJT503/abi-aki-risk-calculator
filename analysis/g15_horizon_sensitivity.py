# -*- coding: utf-8 -*-
"""G15: 视野截断（ascertainment）敏感性分析 — 盲点1 的量化回答
==========================================================
背景：网格止于 min(los−6, 168)；t+48 超出观察窗的未观察结局记阴性。
     信息性截断：LOS 短的患者更少事件 → 截断检查点的事件率被低估/标签偏阴性。
口径（2026-09-23 锁定）：
  完整视野     t + 48 ≤ min(los, 168)
  LOS 截断     t > los − 48            （预测窗越过住院终点）
  窗上限截断   t ≤ los − 48 且 t > 120 （预测窗越过 168h 分析窗上限）
输出（M4 内验 + eICU 外验）：
  ① 三类分解计数（含正/负检查点分别的截断率——信息性截断证据）
  ② 完整视野子集 vs 截断子集的 AUROC/AUPRC/事件率
  ③ 完整视野子集 AUROC 的 stay 聚类 bootstrap 95% CI（1000 次）
门禁：
  - eICU LOS 截断计数必须 = 33,489（2026-09-23 已验证值），偏差>0.1% → stop
  - eICU 三类分解完整占比必须 ≈ 50.0%（±0.5pp）
数据真实性：全部数字来自本脚本实跑输出，落盘 g15_horizon_sensitivity.json
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

SEED = 42
R = "E:/TBI subtype/09_tbi_aki/results"


def classify_visibility(df, los_col="los_h"):
    """t+48 ≤ min(los,168) → complete; t > los−48 → los_trunc; else win_trunc"""
    horizon_end = np.minimum(df[los_col], 168.0)
    complete = df.t_hr + 48.0 <= horizon_end
    los_trunc = df.t_hr > df[los_col] - 48.0
    win_trunc = (~complete) & (~los_trunc)
    return complete, los_trunc, win_trunc


def metrics_block(y, p):
    return {"n": int(len(y)), "pos": int(y.sum()),
            "event_rate": round(float(y.mean()), 4) if len(y) else None,
            "auroc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
            "auprc": round(float(average_precision_score(y, p)), 4) if len(set(y)) > 1 else None}


def stay_cluster_ci(d, y, p, n_boot=1000):
    rng = np.random.default_rng(SEED)
    stays = d.stay_id.unique()
    idx_by_stay = {s: g.index.to_numpy() for s, g in d.groupby("stay_id")}
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(stays, size=len(stays), replace=True)
        idx = np.concatenate([idx_by_stay[s] for s in pick])
        yy, pp = y[idx], p[idx]
        if len(set(yy)) > 1:
            aucs.append(roc_auc_score(yy, pp))
    return {"ci95": [round(float(np.percentile(aucs, 2.5)), 4),
                     round(float(np.percentile(aucs, 97.5)), 4)],
            "n_resamples_valid": len(aucs)}


out = {"probe": "G15 horizon/ascertainment sensitivity", "date": "2026-09-23"}

# ================= eICU 外验 =================
ex = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")
ax_e = pd.read_parquet(f"{R}/n8_3_eicu_abi_event_axis.parquet")[["stay_id", "los_h"]]
ex = ex.merge(ax_e, on="stay_id", how="left")
assert ex.los_h.notna().all(), "eICU los_h 缺失"

comp, lt, wt = classify_visibility(ex)
y, p = ex.y.values, ex.p.values
n_lt_e = int(lt.sum())
rate_complete = float(comp.mean())
print(f"[eICU] complete {comp.sum()} ({rate_complete:.1%}) | los_trunc {lt.sum()} ({lt.mean():.1%}) | win_trunc {wt.sum()} ({wt.mean():.1%})")
if abs(n_lt_e - 33489) / 33489 > 0.001:
    raise SystemExit(f"GATE FAIL: eICU LOS 截断计数 {n_lt_e} ≠ 33489")
if abs(rate_complete - 0.500) > 0.005:
    raise SystemExit(f"GATE FAIL: eICU 完整占比 {rate_complete:.3f} 偏离 50.0% 超容差")

d_ex = ex.reset_index(drop=True)
pos_mask, neg_mask = y == 1, y == 0
res_e = {
    "n_ckpt": int(len(ex)),
    "decomposition": {
        "complete": {"n": int(comp.sum()), "pct": round(rate_complete, 4),
                     "pct_among_pos": round(float(comp[pos_mask].mean()), 4),
                     "pct_among_neg": round(float(comp[neg_mask].mean()), 4)},
        "los_truncated": {"n": n_lt_e, "pct": round(float(lt.mean()), 4),
                          "pct_among_pos": round(float(lt[pos_mask].mean()), 4),
                          "pct_among_neg": round(float(lt[neg_mask].mean()), 4)},
        "window_capped": {"n": int(wt.sum()), "pct": round(float(wt.mean()), 4),
                          "pct_among_pos": round(float(wt[pos_mask].mean()), 4),
                          "pct_among_neg": round(float(wt[neg_mask].mean()), 4)},
    },
    "overall": metrics_block(y, p),
    "complete_only": metrics_block(y[comp.values], p[comp.values]),
    "truncated_only": metrics_block(y[(lt | wt).values], p[(lt | wt).values]),
}
res_e["complete_only_stay_cluster_ci"] = stay_cluster_ci(d_ex[comp.values].reset_index(drop=True),
                                                        y[comp.values], p[comp.values])
out["eICU_external_validation"] = res_e

# ================= M4 内验 =================
iv = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet")
coh_m4 = pd.read_parquet(f"{R}/n1_m4_abi_cohort.parquet")
los_m4 = ((coh_m4.outtime - coh_m4.intime).dt.total_seconds() / 3600.0).rename("los_h")
iv = iv.merge(pd.concat([coh_m4.stay_id, los_m4], axis=1), on="stay_id", how="left")
assert iv.los_h.notna().all(), "M4 los_h 缺失"

comp2, lt2, wt2 = classify_visibility(iv)
y2, p2 = iv.y.values, iv.p.values
print(f"[M4]  complete {comp2.sum()} ({comp2.mean():.1%}) | los_trunc {lt2.sum()} ({lt2.mean():.1%}) | win_trunc {wt2.sum()} ({wt2.mean():.1%})")

d_m4 = iv.reset_index(drop=True)
pos2, neg2 = y2 == 1, y2 == 0
res_m = {
    "n_ckpt": int(len(iv)),
    "decomposition": {
        "complete": {"n": int(comp2.sum()), "pct": round(float(comp2.mean()), 4),
                     "pct_among_pos": round(float(comp2[pos2].mean()), 4),
                     "pct_among_neg": round(float(comp2[neg2].mean()), 4)},
        "los_truncated": {"n": int(lt2.sum()), "pct": round(float(lt2.mean()), 4),
                          "pct_among_pos": round(float(lt2[pos2].mean()), 4),
                          "pct_among_neg": round(float(lt2[neg2].mean()), 4)},
        "window_capped": {"n": int(wt2.sum()), "pct": round(float(wt2.mean()), 4),
                          "pct_among_pos": round(float(wt2[pos2].mean()), 4),
                          "pct_among_neg": round(float(wt2[neg2].mean()), 4)},
    },
    "overall": metrics_block(y2, p2),
    "complete_only": metrics_block(y2[comp2.values], p2[comp2.values]),
    "truncated_only": metrics_block(y2[(lt2 | wt2).values], p2[(lt2 | wt2).values]),
}
res_m["complete_only_stay_cluster_ci"] = stay_cluster_ci(d_m4[comp2.values].reset_index(drop=True),
                                                        y2[comp2.values], p2[comp2.values])
out["M4_internal_validation"] = res_m

with open(f"{R}/g15_horizon_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("\n=== G15 完成 → results/g15_horizon_sensitivity.json ===")
print(json.dumps({"eICU": {"complete": res_e["complete_only"], "ci": res_e["complete_only_stay_cluster_ci"]["ci95"],
                           "overall_auroc": res_e["overall"]["auroc"]},
                  "M4": {"complete": res_m["complete_only"], "ci": res_m["complete_only_stay_cluster_ci"]["ci95"],
                         "overall_auroc": res_m["overall"]["auroc"]}}, indent=1))
