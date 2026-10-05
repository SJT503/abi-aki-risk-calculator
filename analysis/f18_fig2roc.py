# -*- coding: utf-8 -*-
"""f18: regenerate Fig 2 with ROC + PR panels (row 2) added to the existing
statistical panels (row 1, code identical to figures_r11.py).
Row 1: a endpoint-x-cohort AUROC | b paired differences | c severity gradient
Row 2: d ROC Tier-1 (+M4 LR) | e ROC Tier-2 | f PR Tier-1 (+LR+prevalence) | g PR Tier-2
Inputs: package data/f12_full_suite.json + data/f17_pr_data.json (frozen)
Output: package figures/Fig2.{png,pdf,svg} via figstyle.save (style guard rails on)
"""
import os, sys
from pathlib import Path

PKG = Path("E:/TBI subtype/09_tbi_aki/_biomni_r112/final")
sys.path.insert(0, str(PKG / "scripts"))
os.environ["ABI_OUT"] = str(PKG / "figures")
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

F12 = json.load(open(PKG / "data/f12_full_suite.json"))
F17 = json.load(open(PKG / "data/f17_pr_data.json"))
DSC = F12["discrimination"]

CJ, CM, CE = fs.ORANGE, fs.M4, fs.EI
COL = {"int": CJ, "M4": CM, "eICU": CE}
MK_ = {"int": "s", "M4": "o", "eICU": "^"}
NAME = {"int": "JinhuaNSICU internal test", "M4": "MIMIC-IV external", "eICU": "eICU-CRD external"}
EN, MN = "\u2013", "\u2212"

def legend_cohorts(ax, **kw):
    hs = [Line2D([], [], marker=MK_[c], color=COL[c], lw=0, ms=fs.MS, label=NAME[c]) for c in ("int", "M4", "eICU")]
    return ax.legend(handles=hs, **kw)

fig = plt.figure(figsize=(fs.W2, 137 * fs.MM))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.55, left=0.07, right=0.982, bottom=0.09, top=0.93)
g0 = gs[0].subgridspec(1, 3, width_ratios=[1.25, 1.05, 0.9], wspace=0.45)
g1 = gs[1].subgridspec(1, 4, wspace=0.52)

# ---------------- row 1 (verbatim from figures_r11.py) ----------------
ax = fig.add_subplot(g0[0, 0])
eps = [(1, "Tier 1\n\u2265Stage 1"), (2, "\u2265Stage 2\n(merger test)"), (3, "Tier 2\n\u2265Stage 3")]
offs = {"int": -0.24, "M4": 0.0, "eICU": 0.24}
for g, _ in eps:
    for pf in ("int", "M4", "eICU"):
        d = DSC[f"{pf}_ge{g}"]; k = g - 1
        under = (pf == "int" and g >= 2)
        ax.errorbar(k + offs[pf], d["auroc"], yerr=[[d["auroc"] - d["ci"][0]], [min(d["ci"][1], 1.0) - d["auroc"]]], fmt=MK_[pf],
                    color=COL[pf], mfc="white" if under else COL[pf], ms=fs.MS, capsize=fs.CAP, elinewidth=fs.ELW, lw=0)
        ax.text(k + offs[pf], 0.505, f"{d['event_stays']}", fontsize=6, color=COL[pf], ha="center", va="bottom")
ax.text(-0.47, 0.505, "n", fontsize=6, color=fs.GREY, ha="center", va="bottom", style="italic")
ax.set_xticks(range(3)); ax.set_xticklabels([e[1] for e in eps], fontsize=6.5)
ax.set_xlim(-0.55, 2.45); ax.set_ylim(0.5, 1.02); ax.set_ylabel("AUROC (95% CI)")
ax.axhline(1.0, color=fs.LGREY, lw=0.5, ls=":")
legend_cohorts(ax, loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=6, handletextpad=0.1, borderaxespad=0.2)
ax.set_title("Discrimination by endpoint; n = event stays;\nopen symbols: \u22647 event stays (underpowered)", fontsize=6.8, pad=3)
fs.panel(ax, "a", x=-0.2, y=1.12)

ax = fig.add_subplot(g0[0, 1])
comps = [("31", "Tier 2 \u2212 Tier 1"), ("32", "Tier 2 \u2212 \u2265Stage 2"), ("21", "\u2265Stage 2 \u2212 Tier 1")]
yo = {"int": 0.22, "M4": 0.0, "eICU": -0.22}
for i, (key, lab) in enumerate(comps):
    for pf in ("int", "M4", "eICU"):
        v = F12["paired"][f"{pf}_ge{key[0]}_minus_ge{key[1]}"]
        y = len(comps) - 1 - i + yo[pf]
        ax.errorbar(v["delta"], y, xerr=[[v["delta"] - v["ci"][0]], [v["ci"][1] - v["delta"]]], fmt=MK_[pf], color=COL[pf],
                    mfc="white" if pf == "int" else COL[pf], ms=fs.MS, capsize=1.8, elinewidth=fs.ELW, lw=0)
ax.axvline(0, color=fs.GREY, lw=0.7, ls="--")
ax.set_yticks(range(len(comps))); ax.set_yticklabels([c_[1] for c_ in comps][::-1], fontsize=6.5)
ax.set_xlabel("Paired \u0394AUROC (95% CI)"); ax.set_xlim(-0.12, 0.3); ax.set_ylim(-0.6, 2.6)
ax.set_title("Paired stay-level bootstrap\n(1,000 resamples)", fontsize=6.8, pad=3)
fs.panel(ax, "b", x=-0.42, y=1.12)

ax = fig.add_subplot(g0[0, 2])
cats = [("any", "All events\n(Tier 1)"), ("s12", "Stage 1" + EN + "2\nonly"), ("s3", "Stage 3")]
for pf, off in (("M4", -0.14), ("eICU", 0.14)):
    for k, (key, _) in enumerate(cats):
        v = DSC[f"{pf}_grad_{key}"]
        ax.errorbar(k + off, v["auroc"], yerr=[[v["auroc"] - v["ci"][0]], [v["ci"][1] - v["auroc"]]], fmt=MK_[pf], color=COL[pf], ms=fs.MS,
                    capsize=fs.CAP, elinewidth=fs.ELW, lw=0)
        ax.text(k + (-0.03 if off < 0 else 0.03), 0.655, f"{v['n_stays']}", fontsize=6, color=COL[pf], ha="right" if off < 0 else "left", va="bottom")
ax.text(-0.6, 0.655, "n", fontsize=6, color=fs.GREY, ha="left", va="bottom", style="italic")
ax.set_xticks(range(3)); ax.set_xticklabels([c_[1] for c_ in cats], fontsize=6.5)
ax.set_xlim(-0.62, 2.5); ax.set_ylim(0.65, 1.0); ax.set_ylabel("AUROC of the Tier-1 model")
ax.set_title("Tier-1 model, events\nstratified by maximum stage", fontsize=6.8, pad=3)
ax.legend(handles=[Line2D([], [], marker=MK_[c_], color=COL[c_], lw=0, ms=fs.MS, label=NAME[c_].replace(" external", "")) for c_ in ("M4", "eICU")],
          loc="upper left", fontsize=6, handletextpad=0.1, borderaxespad=0.2)
fs.panel(ax, "c", x=-0.25, y=1.12)

# ---------------- row 2: ROC + PR (new) ----------------
def roc_panel(ax, tier):
    for pf in ("int", "M4", "eICU"):
        c = F17[f"{pf}_ge{tier}"]["roc"]
        ax.plot(c["fpr"], c["tpr"], color=COL[pf], lw=0.9, label=NAME[pf].replace(" external", "").replace(" internal test", ""))
    ax.plot([0, 1], [0, 1], color=fs.LGREY, lw=0.6, ls="--", zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")

ax = fig.add_subplot(g1[0, 0])
roc_panel(ax, 1)
lr = F17["M4_ge1_LR"]["roc"]
ax.plot(lr["fpr"], lr["tpr"], color=fs.GREY, lw=0.9, ls=(0, (3, 2)),
        label=f"LR (MIMIC-IV) {F17['M4_ge1_LR']['auroc_check']:.3f}")
ax.legend(fontsize=6.0, loc="lower right", handletextpad=0.3, borderaxespad=0.3, handlelength=1.6)
ax.set_title("ROC, Tier 1 (any AKI)", fontsize=6.8, pad=3)
fs.panel(ax, "d", x=-0.22, y=1.10)

ax = fig.add_subplot(g1[0, 1])
roc_panel(ax, 3)
handles, labels = ax.get_legend_handles_labels()
handles = [Line2D([], [], marker=MK_[c], color=COL[c], lw=0, ms=fs.MS, label=NAME[c].replace(" external", "").replace(" internal test", "")) for c in ("int", "M4", "eICU")]
ax.legend(handles=handles, fontsize=6.0, loc="lower right", handletextpad=0.3, borderaxespad=0.3)
ax.set_title("ROC, Tier 2 (severe AKI)\u2020", fontsize=6.8, pad=3)
fs.panel(ax, "e", x=-0.22, y=1.10)

def pr_panel(ax, tier):
    for pf in ("int", "M4", "eICU"):
        c = F17[f"{pf}_ge{tier}"]
        ax.plot(c["curve"]["recall"], c["curve"]["precision"], color=COL[pf], lw=0.9,
                label=f"{NAME[pf].replace(' external','').replace(' internal test','')} {c['auprc']:.3f}")
        ax.axhline(c["prevalence"], color=COL[pf], lw=0.5, ls=":", alpha=0.7, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")

ax = fig.add_subplot(g1[0, 2])
pr_panel(ax, 1)
lr = F17["M4_ge1_LR"]
ax.plot(lr["curve"]["recall"], lr["curve"]["precision"], color=fs.GREY, lw=0.9, ls=(0, (3, 2)),
        label=f"LR (MIMIC-IV) {lr['auprc']:.3f}")
ax.axhline(lr["prevalence"], color=fs.GREY, lw=0.5, ls=":", alpha=0.7, zorder=0)
ax.legend(fontsize=6.0, loc="upper right", handletextpad=0.3, borderaxespad=0.3, handlelength=1.6)
ax.set_title("PR, Tier 1", fontsize=6.8, pad=3)
fs.panel(ax, "f", x=-0.22, y=1.10)

ax = fig.add_subplot(g1[0, 3])
pr_panel(ax, 3)
ax.legend(fontsize=6.0, loc="upper right", handletextpad=0.3, borderaxespad=0.3, handlelength=1.6)
ax.set_title("PR, Tier 2†", fontsize=6.8, pad=3)
fs.panel(ax, "g", x=-0.22, y=1.10)

# fix title of panel d (AUPRC belongs to PR panels)
fig.axes[3].set_title("ROC, Tier 1 (any AKI)", fontsize=6.8, pad=3)

fs.save(fig, "Fig2")
print("Fig2 regenerated with ROC+PR row")
