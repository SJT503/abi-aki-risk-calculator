# -*- coding: utf-8 -*-
"""G17: KDIGO 阈值规则基线 vs 模型（A4 配敏比较）
==========================================================
规则定义（2026-09-23 从会话转录找回的预演原定义，逐字保留）：
  rule = (uo_ml_kg_h_6h < 0.5) | (cr_last >= 1.5*cr_base) | (cr_last - cr_base >= 0.3)
  —— 当前 6h 尿率 <0.5 mL/kg/h，或当前 Cr 已达 1.5×基线，或较基线升 ≥0.3 mg/dL
  （即"当前证据已逼近/达到 KDIGO 阈值即报警"的朴素规则，非预测模型）
预演复现门禁（M4 全网格 75,638）：TP 1239 / FP 8647 / FN 5582 / TN 60170
  → sens 0.182 / PPV 0.125；任一不符 → stop
输出：
  ① M4 全网格复现（门禁）
  ② M4 内验 + eICU 外验：规则检查点级 sens/PPV/NNE + stay 级捕获/中位 lead
  ③ 模型配敏比较：满阈值扫描（重校准概率），在规则 sens 处线性内插模型 PPV
     + 该 sens 处模型阈值与 stay 级 lead
数据真实性：全部数字来自本脚本实跑输出，落盘 g17_rule_baseline.json
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

from sklearn.metrics import roc_auc_score

SEED = 42
R = "E:/TBI subtype/09_tbi_aki/results"
A_FULL, B_FULL = -0.5414, 0.3569  # 冻结 select-fit 重校准层（g2 复现值）

NEED = ["uo_ml_kg_h_6h", "cr_last", "cr_base"]


def apply_recal(p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return 1 / (1 + np.exp(-(A_FULL + B_FULL * z)))


def rule_flag(df):
    return ((df.uo_ml_kg_h_6h < 0.5) | (df.cr_last >= 1.5 * df.cr_base)
            | ((df.cr_last - df.cr_base) >= 0.3)).astype(int)


def ckpt_metrics(y, flag):
    tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); ppv = tp / max(tp + fp, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": round(sens, 4), "ppv": round(ppv, 4),
            "nne": round(1 / ppv, 2) if ppv > 0 else None,
            "alert_rate": round(float((flag == 1).mean()), 4)}


def stay_lead(df, flag_col, thr_desc):
    """事件 stay 的首次报警 lead（网格行天然 t < onset）"""
    ev = df[(df.onset_hr == df.onset_hr) & (df.y == 1)]
    firsts, captured = [], 0
    for sid, g in ev.groupby("stay_id"):
        f = g[g[flag_col] == 1].sort_values("t_hr")
        if len(f):
            captured += 1
            firsts.append(float(g.onset_hr.iloc[0] - f.t_hr.iloc[0]))
    n = len(ev)
    return {"n_event_stays": int(n), "captured": int(captured),
            "capture_rate": round(captured / max(n, 1), 4),
            "lead_median_h": round(float(np.median(firsts)), 1) if firsts else None,
            "lead_iqr_h": [round(float(np.percentile(firsts, 25)), 1),
                           round(float(np.percentile(firsts, 75)), 1)] if firsts else None,
            "def": thr_desc}


def sweep_matched_sens(y, p_cal, sens_target):
    """满阈值扫描 + 在 sens_target 处线性内插 PPV/阈值"""
    qs = np.quantile(p_cal, np.linspace(0.001, 0.999, 2000))
    thr = np.unique(qs)
    sens_arr, ppv_arr, thr_arr = [], [], []
    order = np.argsort(-p_cal)  # 降序
    ys = y[order]
    tp_c = np.cumsum(ys == 1); fp_c = np.cumsum(ys == 0)
    P = int((y == 1).sum())
    for t in thr:
        k = int((p_cal >= t).sum())
        if k == 0:
            continue
        tp, fp = tp_c[k - 1], fp_c[k - 1]
        sens_arr.append(tp / P); ppv_arr.append(tp / max(tp + fp, 1)); thr_arr.append(t)
    sens_arr = np.array(sens_arr); ppv_arr = np.array(ppv_arr); thr_arr = np.array(thr_arr)
    if sens_target > sens_arr.max() or sens_target < sens_arr.min():
        return None
    ppv_interp = float(np.interp(sens_target, sens_arr[::-1], ppv_arr[::-1]))
    thr_interp = float(np.interp(sens_target, sens_arr[::-1], thr_arr[::-1]))
    return {"sens_matched": round(sens_target, 4),
            "model_ppv_interpolated": round(ppv_interp, 4),
            "model_nne": round(1 / ppv_interp, 2),
            "model_threshold_interpolated": round(thr_interp, 4),
            "sens_range_covered": [round(float(sens_arr.min()), 4), round(float(sens_arr.max()), 4)]}


def model_at(y, p_cal, tau):
    flag = (p_cal >= tau).astype(int)
    return ckpt_metrics(y, flag)


out = {"probe": "G17 KDIGO-threshold rule baseline vs model (matched sensitivity)", "date": "2026-09-23",
       "rule_definition": "uo_ml_kg_h_6h < 0.5 OR cr_last >= 1.5*cr_base OR (cr_last - cr_base) >= 0.3",
       "recal_layer": {"a": A_FULL, "b": B_FULL}}

# ================= ① M4 全网格复现门禁 =================
F = pd.read_parquet(f"{R}/n5_abi_rolling_features.parquet")
g_full = pd.read_parquet(f"{R}/n3_m4_abi_grid.parquet")
m_full = g_full.merge(F[NEED + ["stay_id", "t_hr"]].drop_duplicates(["stay_id", "t_hr"]),
                      on=["stay_id", "t_hr"], how="left")
flag_full = rule_flag(m_full)
rep = ckpt_metrics(m_full.label.values, flag_full.values)
print(f"[gate·复现] M4全网格: TP {rep['tp']} FP {rep['fp']} FN {rep['fn']} TN {rep['tn']} "
      f"sens {rep['sensitivity']} ppv {rep['ppv']} (期望 TP1239 FP8647 FN5582 TN60170 / 0.182 / 0.125)")
if (rep["tp"], rep["fp"], rep["fn"], rep["tn"]) != (1239, 8647, 5582, 60170):
    raise SystemExit(f"GATE FAIL: M4 全网格复现失败 {rep}")
out["m4_full_grid_reproduction"] = rep

# ================= ② M4 内验 =================
iv = pd.read_parquet(f"{R}/n6_abi_gbm_preds.parquet")
iv = iv.merge(F[NEED + ["stay_id", "t_hr"]].drop_duplicates(["stay_id", "t_hr"]),
              on=["stay_id", "t_hr"], how="left")
iv = iv.merge(g_full[["stay_id", "onset_hr"]].drop_duplicates("stay_id"), on="stay_id", how="left")
iv["rule"] = rule_flag(iv)
iv["p_cal"] = apply_recal(iv.p.values)

rule_m4 = ckpt_metrics(iv.y.values, iv.rule.values)
lead_rule_m4 = stay_lead(iv, "rule", "rule baseline")
sens_t = rule_m4["sensitivity"]
match_m4 = sweep_matched_sens(iv.y.values, iv.p_cal.values, sens_t)
iv[f"p_ge"] = (iv.p_cal >= match_m4["model_threshold_interpolated"]).astype(int) if match_m4 else 0
lead_mod_m4 = stay_lead(iv, "p_ge", f"model @ matched sens {sens_t:.3f}") if match_m4 else None
out["M4_internal_validation"] = {
    "rule": rule_m4, "rule_stay_lead": lead_rule_m4,
    "model_matched_sens": match_m4, "model_matched_stay_lead": lead_mod_m4,
    "model_at_020": model_at(iv.y.values, iv.p_cal.values, 0.20),
    "model_at_010": model_at(iv.y.values, iv.p_cal.values, 0.10),
}
print(f"[M4内验] rule: sens {sens_t} ppv {rule_m4['ppv']} nne {rule_m4['nne']} | "
      f"model matched: ppv {match_m4['model_ppv_interpolated']} nne {match_m4['model_nne']}")

# ================= ③ eICU 外验 =================
E = pd.read_parquet(f"{R}/n8_4_eicu_abi_features.parquet")
missing = [c for c in NEED if c not in E.columns]
if missing:
    raise SystemExit(f"GATE FAIL: eICU 特征缺列 {missing}")
ex = pd.read_parquet(f"{R}/n8_5_eicu_abi_external.parquet")
ex = ex.merge(E[NEED + ["stay_id", "t_hr"]].drop_duplicates(["stay_id", "t_hr"]),
              on=["stay_id", "t_hr"], how="left")
g_e = pd.read_parquet(f"{R}/n8_3_eicu_abi_grid.parquet")
ex = ex.merge(g_e[["stay_id", "onset_hr"]].drop_duplicates("stay_id"), on="stay_id", how="left")
ex["rule"] = rule_flag(ex)
ex["p_cal"] = apply_recal(ex.p.values)

rule_ex = ckpt_metrics(ex.y.values, ex.rule.values)
lead_rule_ex = stay_lead(ex, "rule", "rule baseline")
sens_e = rule_ex["sensitivity"]
match_ex = sweep_matched_sens(ex.y.values, ex.p_cal.values, sens_e)
ex["p_ge"] = (ex.p_cal >= match_ex["model_threshold_interpolated"]).astype(int) if match_ex else 0
lead_mod_ex = stay_lead(ex, "p_ge", f"model @ matched sens {sens_e:.3f}") if match_ex else None
out["eICU_external_validation"] = {
    "rule": rule_ex, "rule_stay_lead": lead_rule_ex,
    "model_matched_sens": match_ex, "model_matched_stay_lead": lead_mod_ex,
    "model_at_020": model_at(ex.y.values, ex.p_cal.values, 0.20),
    "model_at_010": model_at(ex.y.values, ex.p_cal.values, 0.10),
}
print(f"[eICU] rule: sens {sens_e} ppv {rule_ex['ppv']} nne {rule_ex['nne']} | "
      f"model matched: ppv {match_ex['model_ppv_interpolated']} nne {match_ex['model_nne']}")

with open(f"{R}/g17_rule_baseline.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("\n=== G17 完成 → results/g17_rule_baseline.json ===")
