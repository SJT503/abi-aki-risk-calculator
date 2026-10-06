"""Round 11 (Paper 2) figures: Fig1-4, SuppFig1-4 (SuppFig3 hospital plot and SuppFig4 SHAP beeswarm from the supplied f12 SHAP matrix and hospital file, read from PRIVATE_DIR and not shipped). Every plotted number is read from data/ (JSON) via numbers_r11.load()
(companion-package cohort descriptors only for the funnel of the external cohorts).
Usage: PANEL_CASE=lower python figures_r11.py DATA COMPANION OUT STYLE_DIR PRIVATE_DIR
Colour / marker mapping (fixed across all figures): JinhuaNSICU development + internal test = orange square;
MIMIC-IV external = teal circle; eICU-CRD external = red triangle.
"""
import os, sys
DATA, DC, OUT, STYLE = sys.argv[1:5]
os.environ['ABI_DATA'] = DATA; os.environ['ABI_OUT'] = OUT
assert os.environ.get('PANEL_CASE') == 'lower', 'set PANEL_CASE=lower'
sys.path.insert(0, STYLE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import numbers_r11 as nr
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import logging; logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

r3 = fs.r3
N, LEDGER, X = nr.load(DATA, DC)
F10, F11, F12, DSC, CAL, DEP = X['F10'], X['F11'], X['F12'], X['DSC'], X['F12']['calibration'], X['F12']['deployment']

# === readable feature labels (f21) ===
import json
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)


# === readable feature labels (f21) ===
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)


# === readable feature labels (f21) ===
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)


# === readable feature labels (f21) ===
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)


# === readable feature labels (f21) ===
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)


# === readable feature labels (f21) ===
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)


# === readable feature labels (f21) ===
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)


# === readable feature labels (f21) ===
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)


# === readable feature labels (f21) ===
try:
    _LBL = json.load(open(os.path.join(DATA, 'f21_feature_labels.json')))
except FileNotFoundError:
    _LBL = {}
def rlabel(f):
    return _LBL.get(f, f)

LOG = []
CJ, CM, CE = fs.ORANGE, fs.M4, fs.EI                      # Jinhua / MIMIC-IV / eICU
COL = {'int': CJ, 'M4': CM, 'eICU': CE}
MK_ = {'int': 's', 'M4': 'o', 'eICU': '^'}
NAME = {'int': 'JinhuaNSICU internal test', 'M4': 'MIMIC-IV external', 'eICU': 'eICU-CRD external'}
FILL = {'int': '#FDF1DC', 'M4': '#E7F1F4', 'eICU': '#FBEAEA'}
nfmt = lambda x: f'{int(x):,}'
EN, MN = '\u2013', '\u2212'


def legend_cohorts(ax, **kw):
    hs = [Line2D([], [], marker=MK_[c], color=COL[c], lw=0, ms=fs.MS, label=NAME[c]) for c in ('int', 'M4', 'eICU')]
    return ax.legend(handles=hs, **kw)


# ============================ Fig 1: cohorts + pipeline (vector) ============================
fig = plt.figure(figsize=(fs.W2, 125 * fs.MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 125); ax.axis('off')
boxes, free = [], []


def box(x, y, w, h, lines, ec, fc='white', lab=None, fsz=6.3, bold_first=True, lw=0.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', ec=ec, fc=fc, lw=lw)
    ax.add_patch(p)
    ts = []
    n_ = len(lines); lh = min(3.3, (h - 1.6) / max(n_, 1))
    y0 = y + h / 2 + lh * (n_ - 1) / 2
    for i, s in enumerate(lines):
        ts.append(ax.text(x + w / 2, y0 - i * lh, s, ha='center', va='center', fontsize=fsz,
                          fontweight='bold' if (bold_first and i == 0) else 'normal', color=fs.INK))
    boxes.append((lab or lines[0], p, ts))
    return p


def arrow(x0, y0, x1, y1, col=fs.GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=6, lw=0.7, color=col, shrinkA=0, shrinkB=0))


fs.panel(ax, 'a', x=0.5, y=124.5, transform=ax.transData, va='top')
XJ, WJ, XM, WM, XE, WE = 1, 64, 70, 52, 127, 52
hdr = [ax.text(XJ + WJ / 2, 122.5, 'JinhuaNSICU (development)', ha='center', va='top', fontsize=7, fontweight='bold', color=CJ),
       ax.text(XM + WM / 2, 122.5, 'MIMIC-IV v3.1 (external)', ha='center', va='top', fontsize=7, fontweight='bold', color=CM),
       ax.text(XE + WE / 2, 122.5, 'eICU-CRD v2.0 (external)', ha='center', va='top', fontsize=7, fontweight='bold', color=CE)]
free += [('hJ', hdr[0]), ('hM', hdr[1]), ('hE', hdr[2])]
Y1, Y2, Y3, H1, H3 = 103, 87, 66, 11, 17
c = X['cohort']
box(XJ, Y1, WJ, H1, ['Stays with a valid checkpoint grid', f"{N['jh_stays']} stays; {N['jh_subjects']} subjects"], CJ, lab='J1')
box(XJ, Y2, WJ, H1, ['Split by subject, 80% / 20%', 'fixed seed; subject overlap 0'], CJ, lab='J2')
box(XJ, Y3, 31, H3, ['Development', f"{N['jh_train_subj']} subjects, {N['t1_dev_stays']} stays", f"{N['t1_dev_ckpt']} checkpoints",
                     f"Positive: T1 {N['t1_dev_ge1_ckpt']}, T2 {N['t1_dev_ge3_ckpt']}"], CJ, fc=FILL['int'], lab='dev')
box(XJ + 33, Y3, 31, H3, ['Internal test', f"{N['jh_test_subj']} subjects, {N['t1_test_stays']} stays", f"{N['t1_test_ckpt']} checkpoints",
                          f"Event stays T1 {N['t1_test_ge1_stays']}, T2 {N['t1_test_ge3_stays']}"], CJ, fc=FILL['int'], lab='tst')
box(XM, Y1, WM, H1, ['ICU stays, acute brain injury', f"{N['m4_stays']} stays"], CM, lab='M1')
box(XM, Y2, WM, H1, ['At risk at the 24-h landmark', f"{N['m4_atrisk']} stays"], CM, lab='M2')
box(XM, Y3, WM, H3, ['Checkpoint grid, 24' + EN + '168 h', f"{N['t1_m4_stays']} stays", f"{N['t1_m4_ckpt']} checkpoints",
                     f"Event stays T1 {N['t1_m4_ge1_stays']}, T2 {N['t1_m4_ge3_stays']}"], CM, fc=FILL['M4'], lab='M3')
box(XE, Y1, WE, H1, ['ICU stays (crosswalk cohort)', f"{N['ei_stays_cw']} stays; {N['ei_hosp_all']} hospitals"], CE, lab='E1')
box(XE, Y2, WE, H1, ['At risk at the 24-h landmark', f"{N['ei_atrisk']} stays"], CE, lab='E2')
box(XE, Y3, WE, H3, ['Checkpoint grid, 24' + EN + '168 h', f"{N['t1_ei_stays']} stays; {N['ei_hosp']} hospitals", f"{N['t1_ei_ckpt']} checkpoints",
                     f"Event stays T1 {N['t1_ei_ge1_stays']}, T2 {N['t1_ei_ge3_stays']}"], CE, fc=FILL['eICU'], lab='E3')
for xc, ys in ((XJ + WJ / 2, (Y1, Y2)), (XM + WM / 2, (Y1, Y2)), (XE + WE / 2, (Y1, Y2))):
    arrow(xc, ys[0], xc, ys[1] + H1)
arrow(XJ + 15.5, Y2, XJ + 15.5, Y3 + H3); arrow(XJ + 48.5, Y2, XJ + 48.5, Y3 + H3)
arrow(XM + WM / 2, Y2, XM + WM / 2, Y3 + H3); arrow(XE + WE / 2, Y2, XE + WE / 2, Y3 + H3)
# frozen-model route: development box -> external grids
yr = 58
ax.plot([XJ + 15.5, XJ + 15.5], [Y3, yr], color=fs.GREY, lw=0.7)
ax.plot([XJ + 15.5, XE + WE / 2], [yr, yr], color=fs.GREY, lw=0.7)
arrow(XM + WM / 2, yr, XM + WM / 2, Y3, col=fs.GREY); arrow(XE + WE / 2, yr, XE + WE / 2, Y3, col=fs.GREY)
fz = ax.text(XJ + 34, yr - 3.2, 'Frozen JinhuaNSICU models and any-AKI recalibration layer, applied without refitting', ha='left', va='center',
             fontsize=6, style='italic', color=fs.GREY)
free.append(('frozen', fz))
# panel b: pipeline strip
fs.panel(ax, 'b', x=0.5, y=48.5, transform=ax.transData, va='top')
yp, hp = 14, 26
box(2, yp, 32, hp, ['Checkpoint t', 'every 6 h, 24' + EN + '168 h', 'after ICU admission', 'data: admission \u2192 t only'], fs.INK, lab='ck')
box(39, yp, 38, hp, [f"{N['n_feat']} features", f"from {N['n_cand']} candidates", 'creatinine kinetics, vitals', 'laboratory, acid' + EN + 'base', 'treatments, baseline',
                     'no GCS or urine output'], fs.INK, lab='ft')
box(82, yp, 26, hp, ['LightGBM', 'one model', 'per endpoint', '(class-weighted)'], fs.INK, lab='lgb')
box(113, yp, 25, hp, ['Recalibration', 'logit layer (any AKI)', 'out-of-fold, grouped', 'by subject'], fs.INK, lab='cal')
for x0, x1 in ((34, 39), (77, 82), (108, 113)):
    arrow(x0, yp + hp / 2, x1, yp + hp / 2, col=fs.INK)
box(142, 28, 36, 12, ['Any AKI (\u2265Stage 1)', 'Stages 1 and 2 merged', 'primary outcome'], CM, fc=FILL['M4'], lab='t1')
box(142, 14, 36, 12, ['Severe AKI (\u2265Stage 3)', 'separate model', 'key secondary outcome'], fs.ORANGE, fc='#FDF1DC', lab='t2')
arrow(138, yp + hp / 2, 142, 34, col=fs.INK); arrow(138, yp + hp / 2, 142, 20, col=fs.INK)
wlab = ax.text(90, 8.5, 'Label: highest KDIGO creatinine stage reached in [t, t + 54 h) (current + next eight 6-h bins)', ha='center', va='center', fontsize=6.3, color=fs.INK)
free.append(('window', wlab))
fs.check_boxes(fig, 'Fig1', boxes, free)
LOG.append('Fig1 check_boxes ALL PASS')
fs.save(fig, 'Fig1', align=False)

# ============================ Fig 2: discrimination ============================
fig = plt.figure(figsize=(fs.W2, 145 * fs.MM))
gs0 = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.55)
gs = gs0[0].subgridspec(1, 3, width_ratios=[1.25, 1.05, 0.9], wspace=0.45)
# a: AUROC by endpoint x cohort
ax = fig.add_subplot(gs[0, 0])
eps = [(1, 'Any AKI\n\u2265Stage 1'), (3, 'Severe AKI\n\u2265Stage 3')]
offs = {'int': -0.24, 'M4': 0.0, 'eICU': 0.24}
for gi, (g, _) in enumerate(eps):
    for pf in ('int', 'M4', 'eICU'):
        d = DSC[f'{pf}_ge{g}']; k = gi
        under = (pf == 'int' and g >= 2)
        ax.errorbar(k + offs[pf], d['auroc'], yerr=[[d['auroc'] - d['ci'][0]], [min(d['ci'][1], 1.0) - d['auroc']]], fmt=MK_[pf],
                    color=COL[pf], mfc='white' if under else COL[pf], ms=fs.MS, capsize=fs.CAP, elinewidth=fs.ELW, lw=0)
        ax.text(k + offs[pf], 0.505, f"{d['event_stays']}", fontsize=6, color=COL[pf], ha='center', va='bottom')
ax.text(-0.47, 0.505, 'n', fontsize=6, color=fs.GREY, ha='center', va='bottom', style='italic')
ax.set_xticks(range(len(eps))); ax.set_xticklabels([e[1] for e in eps], fontsize=6.5)
ax.set_xlim(-0.55, len(eps)-0.55); ax.set_ylim(0.5, 1.02); ax.set_ylabel('AUROC (95% CI)')
ax.axhline(1.0, color=fs.LGREY, lw=0.5, ls=':')
legend_cohorts(ax, loc='lower left', bbox_to_anchor=(0.0, 0.07), fontsize=6, handletextpad=0.1, borderaxespad=0.2)
ax.set_title('Discrimination by endpoint; n = event stays;\nopen symbols: \u22647 event stays (underpowered)', fontsize=6.8, pad=3)
fs.panel(ax, 'a', x=-0.2, y=1.12)
# b: paired differences
ax = fig.add_subplot(gs[0, 1])
comps = [('31', 'Severe AKI \u2212 any AKI')]
yo = {'int': 0.22, 'M4': 0.0, 'eICU': -0.22}
for i, (key, lab) in enumerate(comps):
    for pf in ('int', 'M4', 'eICU'):
        v = F12['paired'][f'{pf}_ge{key[0]}_minus_ge{key[1]}']
        y = len(comps) - 1 - i + yo[pf]
        ax.errorbar(v['delta'], y, xerr=[[v['delta'] - v['ci'][0]], [v['ci'][1] - v['delta']]], fmt=MK_[pf], color=COL[pf],
                    mfc='white' if pf == 'int' else COL[pf], ms=fs.MS, capsize=1.8, elinewidth=fs.ELW, lw=0)
ax.axvline(0, color=fs.GREY, lw=0.7, ls='--')
ax.set_yticks(range(len(comps))); ax.set_yticklabels([c_[1] for c_ in comps][::-1], fontsize=6.5)
ax.set_xlabel('Paired \u0394AUROC (95% CI)'); ax.set_xlim(-0.12, 0.3); ax.set_ylim(-0.6, 0.6)
ax.set_title('Paired stay-level bootstrap\n(1,000 resamples)', fontsize=6.8, pad=3)
fs.panel(ax, 'b', x=-0.42, y=1.12)
# c: severity gradient
ax = fig.add_subplot(gs[0, 2])
cats = [('any', 'All events\n(any AKI)'), ('s12', 'Stage 1' + EN + '2\nonly'), ('s3', 'Stage 3')]
for pf, off in (('M4', -0.14), ('eICU', 0.14)):
    for k, (key, _) in enumerate(cats):
        v = DSC[f'{pf}_grad_{key}']
        ax.errorbar(k + off, v['auroc'], yerr=[[v['auroc'] - v['ci'][0]], [v['ci'][1] - v['auroc']]], fmt=MK_[pf], color=COL[pf], ms=fs.MS,
                    capsize=fs.CAP, elinewidth=fs.ELW, lw=0)
        ax.text(k + (-0.03 if off < 0 else 0.03), 0.655, f"{v['n_stays']}", fontsize=6, color=COL[pf], ha='right' if off < 0 else 'left', va='bottom')
ax.text(-0.6, 0.655, 'n', fontsize=6, color=fs.GREY, ha='left', va='bottom', style='italic')
ax.set_xticks(range(3)); ax.set_xticklabels([c_[1] for c_ in cats], fontsize=6.5)
ax.set_xlim(-0.62, 2.5); ax.set_ylim(0.65, 1.0); ax.set_ylabel('AUROC of the any-AKI model')
ax.set_title('Any-AKI model, events\nstratified by maximum stage', fontsize=6.8, pad=3)
ax.legend(handles=[Line2D([], [], marker=MK_[c_], color=COL[c_], lw=0, ms=fs.MS, label=NAME[c_].replace(' external', '')) for c_ in ('M4', 'eICU')],
          loc='upper left', fontsize=6, handletextpad=0.1, borderaxespad=0.2)
fs.panel(ax, 'c', x=-0.25, y=1.12)

# --- row 2: ROC + PR (from f17_pr_data.json, styled as row 1) ---
try:
    F17 = json.load(open(os.path.join(DATA, 'f17_pr_data.json')))
except FileNotFoundError:
    F17 = None
if F17:
    gs2 = gs0[1].subgridspec(1, 4, wspace=0.65)

    def _roc(ax, tier, ylab=True):
        for pf in ('int', 'M4', 'eICU'):
            c = F17[f'{pf}_ge{tier}']
            lbl = NAME[pf].replace(' external', '').replace(' internal test', '')
            lbl += ' ' + format(c.get('auroc_check', 0), '.3f')
            ax.plot(c['roc']['fpr'], c['roc']['tpr'], color=COL[pf], lw=1.2, label=lbl)
        ax.plot([0, 1], [0, 1], color=fs.LGREY, lw=0.6, ls='--', zorder=0)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel('False positive rate', fontsize=7)
        if ylab: ax.set_ylabel('True positive rate', fontsize=7)
        ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs2[0, 0]); _roc(ax, 1)
    ax.legend(fontsize=6.0, loc='lower right', handletextpad=0.3, borderaxespad=0.4, handlelength=1.5)
    ax.set_title('ROC, any AKI', fontsize=7, pad=3)
    fs.panel(ax, 'd', x=-0.22, y=1.12)

    ax = fig.add_subplot(gs2[0, 1]); _roc(ax, 3, ylab=False)
    ax.legend(fontsize=6.0, loc='lower right', handletextpad=0.3, borderaxespad=0.4, handlelength=1.5)
    ax.set_title('ROC, severe AKI†', fontsize=7, pad=3)
    fs.panel(ax, 'e', x=-0.22, y=1.12)

    def _pr(ax, tier, ylab=True):
        for pf in ('int', 'M4', 'eICU'):
            c = F17[f'{pf}_ge{tier}']
            lbl = NAME[pf].replace(' external', '').replace(' internal test', '')
            lbl += ' ' + format(c['auprc'], '.3f')
            ax.plot(c['curve']['recall'], c['curve']['precision'], color=COL[pf], lw=1.2, label=lbl)
            ax.axhline(c['prevalence'], color=COL[pf], lw=0.6, ls=':', alpha=0.8, zorder=0)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
        ax.set_xlabel('Recall', fontsize=7)
        if ylab: ax.set_ylabel('Precision', fontsize=7)
        ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs2[0, 2]); _pr(ax, 1)
    ax.legend(fontsize=6.0, loc='upper right', handletextpad=0.3, borderaxespad=0.4, handlelength=1.5)
    ax.set_title('PR, any AKI', fontsize=7, pad=3)
    fs.panel(ax, 'f', x=-0.22, y=1.12)

    ax = fig.add_subplot(gs2[0, 3]); _pr(ax, 3, ylab=False)
    ax.legend(fontsize=6.0, loc='upper right', handletextpad=0.3, borderaxespad=0.4, handlelength=1.5)
    ax.set_title('PR, severe AKI†', fontsize=7, pad=3)
    fs.panel(ax, 'g', x=-0.22, y=1.12)

fs.save(fig, 'Fig2')

# ============================ Fig 3: calibration + deployment ============================
fig = plt.figure(figsize=(fs.W2, 205 * fs.MM))
gs = fig.add_gridspec(3, 3, wspace=0.45, hspace=0.55, left=0.085, right=0.985, bottom=0.06, top=0.955)
curves = {'int': F11['internal_cal_curve'], 'M4': F11['M4_ext']['cal_curve_ge1'], 'eICU': F11['eICU_ext']['cal_curve_ge1']}
ci_ = {'int': 'int', 'M4': 'M4', 'eICU': 'eICU'}
lim = 0.35
for i, pf in enumerate(('int', 'M4', 'eICU')):
    ax = fig.add_subplot(gs[0, i]); mp, oy = np.array(curves[pf][0]), np.array(curves[pf][1])
    assert max(mp.max(), oy.max()) < lim
    ax.plot([0, lim], [0, lim], color=fs.LGREY, lw=0.6, ls='--', zorder=0)
    ax.plot(mp, oy, MK_[pf], color=COL[pf], ms=fs.MS, lw=0, mfc=COL[pf])
    ax.plot(mp, oy, '-', color=COL[pf], lw=0.6, alpha=0.5)
    c0, c1 = CAL[pf]['raw'], CAL[pf]['recal']
    ax.text(0.04, 0.97, f"Slope {c0['slope']:.3f} \u2192 {c1['slope']:.3f}\nECE {c0['ece']:.4f} \u2192 {c1['ece']:.4f}", transform=ax.transAxes, va='top', fontsize=6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal'); ax.set_xticks([0, .1, .2, .3]); ax.set_yticks([0, .1, .2, .3])
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed event rate' if i == 0 else '')
    ax.set_title(NAME[pf], fontsize=7); fs.panel(ax, 'abc'[i], x=-0.3, y=1.08)
    LOG.append(f'Fig3 {pf}: decile max predicted {mp.max():.4f}, max observed {oy.max():.4f}')
T = ['0.05', '0.08', '0.1', '0.15', '0.2']
# DCA row (PI order 2026-10-06: calibration -> clinical utility -> deployment; formerly SuppFig 2)
dca = {'int': F11['internal_dca'], 'M4': F11['M4_ext']['dca_ge1'], 'eICU': F11['eICU_ext']['dca_ge1']}
for i, pf in enumerate(('int', 'M4', 'eICU')):
    ax = fig.add_subplot(gs[1, i]); d = pd.DataFrame(dca[pf])
    ax.plot(d.thr, d.nb_model, '-' + MK_[pf], color=COL[pf], lw=fs.LW, ms=fs.MS, label='Model')
    ax.plot(d.thr, d.nb_treat_all, '--', color=fs.GREY, lw=0.9, label='Treat all')
    ax.axhline(0, color=fs.INK, lw=0.7, ls=':', label='Treat none')
    ax.axvline(0.15, color=fs.LGREY, lw=0.7, ls='-')
    ax.set_ylim(-0.012, 0.042); ax.set_xlim(0.03, 0.52); ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit' if i == 0 else '')
    ax.set_title(NAME[pf], fontsize=7)
    if i == 0:
        ax.legend(loc='upper right', fontsize=6, handletextpad=0.3, borderaxespad=0.2)
    fs.panel(ax, 'def'[i], x=-0.25 if i == 0 else -0.18, y=1.08)
    LOG.append(f"Fig3 DCA {pf}: last positive-threshold {d.thr[d.nb_model > 0].max() if (d.nb_model > 0).any() else 0}")
ax = fig.add_subplot(gs[2, 0])
for pf in ('int', 'M4', 'eICU'):
    xs = [100 * DEP[pf][f'thr{t}']['n_alert_stays'] / X['STAYS'][pf] for t in T]; ys = [100 * DEP[pf][f'thr{t}']['capture'] for t in T]
    ax.plot(xs, ys, '-', color=COL[pf], lw=0.8, alpha=0.6); ax.plot(xs, ys, MK_[pf], color=COL[pf], ms=fs.MS)
    if pf == 'eICU':
        for t, x_, y_ in zip(T, xs, ys):
            ax.annotate(f'{float(t):.2f}', (x_, y_), xytext=(2, 5) if t == '0.05' else (-3, 5), ha='left' if t == '0.05' else 'right', textcoords='offset points', fontsize=6, color=fs.INK)
ax.set_xlim(0, 85); ax.set_ylim(0, 100); ax.set_xlabel('Stays alerted at least once (%)'); ax.set_ylabel('Event stays captured (%)')
ax.set_title('Workload' + EN + 'capture; labels: threshold', fontsize=6.8); fs.panel(ax, 'g', x=-0.3, y=1.08)
ax = fig.add_subplot(gs[2, 1])
for pf in ('int', 'M4', 'eICU'):
    ys = [DEP[pf][f'thr{t}']['NNE'] for t in T]
    ax.plot([float(t) for t in T], ys, '-', color=COL[pf], lw=0.8, alpha=0.6); ax.plot([float(t) for t in T], ys, MK_[pf], color=COL[pf], ms=fs.MS)
ax.set_xticks([float(t) for t in T]); ax.set_xticklabels(['0.05', '0.08', '0.10', '0.15', '0.20'], fontsize=6.5)
ax.set_ylim(0, 12.5); ax.set_xlabel('Alert threshold'); ax.set_ylabel('NNE (alerted per captured stay)')
ax.set_title('Number needed to evaluate', fontsize=6.8); fs.panel(ax, 'h', x=-0.3, y=1.08)
legend_cohorts(ax, loc='upper right', fontsize=6, handletextpad=0.1, borderaxespad=0.2)
ax = fig.add_subplot(gs[2, 2])
for pf in ('int', 'M4', 'eICU'):
    ys = [DEP[pf][f'thr{t}']['FA_per_100ptd'] for t in T]
    ax.plot([float(t) for t in T], ys, '-', color=COL[pf], lw=0.8, alpha=0.6); ax.plot([float(t) for t in T], ys, MK_[pf], color=COL[pf], ms=fs.MS)
ax.set_xticks([float(t) for t in T]); ax.set_xticklabels(['0.05', '0.08', '0.10', '0.15', '0.20'], fontsize=6.5)
ax.set_ylim(0, 20); ax.set_xlabel('Alert threshold'); ax.set_ylabel('False alarms per 100\nevent-free patient-days')
ax.set_title('False-alarm rate', fontsize=6.8); fs.panel(ax, 'i', x=-0.3, y=1.08)
fs.save(fig, 'Fig3')

# ============================ Fig 4: external deployment profile, trend, subgroups (MIMIC-IV row then eICU-CRD row; PI order 2026-10-05) ============================
F27 = json.load(open(os.path.join(DATA, 'f27_m4_subgroups_mk.json')))   # M4 MK + subgroups (f27; closes the previous eICU-only asymmetry)
fig = plt.figure(figsize=(fs.W2, 158 * fs.MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.0, 1.7], wspace=0.55, hspace=0.72, left=0.085, right=0.995, bottom=0.085, top=0.90)
xlb = ['24' + EN + '48', '48' + EN + '72', '72' + EN + '96', '96' + EN + '120', '120' + EN + '144', '144' + EN + '168']
frows = [('Overall', 'Overall'), ('Age <65', 'Age <65 y'), ('Age >=65', 'Age \u226565 y'), ('Male', 'Male'), ('Female', 'Female'), ('TBI', 'TBI'),
         ('SAH/ICH', 'Hemorrhagic (SAH/ICH)'), ('IS/unspec', 'Ischemic (IS/unspecified)'), ('CKD', 'CKD'), ('No CKD', 'No CKD')]
for r_, (pf, disp, MKd, SUB) in enumerate([('M4', 'MIMIC-IV', F27['mk'], F27['subgroups']),
                                           ('eICU', 'eICU-CRD', F12['mk'], F12['subgroups'])]):
    cc = COL[pf]; mk_ = MK_[pf]
    # lead time profile
    ax = fig.add_subplot(gs[r_, 0])
    xs = np.arange(len(T))
    for k, t in enumerate(T):
        v = DEP[pf][f'thr{t}']
        ax.errorbar(k, v['lead_median_h'], yerr=[[v['lead_median_h'] - v['lead_iqr'][0]], [v['lead_iqr'][1] - v['lead_median_h']]], fmt=mk_, color=cc, ms=fs.MS,
                    capsize=fs.CAP, elinewidth=fs.ELW, lw=0)
        ax.text(k, 62, f"{100 * v['capture']:.1f}%", fontsize=6, ha='center', va='bottom', color=fs.INK)
    ax.set_xticks(xs); ax.set_xticklabels(['0.05', '0.08', '0.10', '0.15', '0.20'], fontsize=6.5)
    ax.set_xlim(-0.6, 4.6); ax.set_ylim(0, 70); ax.set_yticks([0, 12, 24, 36, 48, 60])
    ax.set_xlabel('Alert threshold'); ax.set_ylabel('Lead time, median (IQR) (h)')
    ax.set_title(disp + ': lead time\n(top: % event stays captured)', fontsize=6.8, pad=3); fs.panel(ax, 'ad'[r_], x=-0.42, y=1.2)
    LOG.append(f"Fig4 {pf} lead: " + '; '.join(f"thr{t}={DEP[pf][f'thr{t}']['lead_median_h']}h" for t in T))
    # MK strata
    ax = fig.add_subplot(gs[r_, 1])
    pooled = DSC[f'{pf}_ge1']['auroc']
    ax.plot(range(6), MKd['strata'], mk_ + '-', color=cc, ms=fs.MS, lw=fs.LW)
    prev_side, prev_v, prev_deep = None, None, False
    for i, v in enumerate(MKd['strata']):
        up = v > pooled   # label on the far side of the point from the pooled-AUROC line
        side = 'up' if up else 'dn'
        # stagger the depth when the previous label sits on the same side at nearly the same height,
        # so horizontally adjacent same-side labels (e.g. M4 0.764 / 0.761) never touch
        deep = (side == prev_side) and (prev_v is not None) and abs(v - prev_v) < 0.006 and not prev_deep
        dy = 0.006 if up else -(0.012 if not deep else 0.022)
        ax.text(i + (0.08 if not up else -0.08), v + dy, r3(v), ha='left' if not up else 'right', va='top' if not up else 'bottom', fontsize=6, color=cc)
        prev_side, prev_v, prev_deep = side, v, deep
    ax.axhline(pooled, color=fs.GREY, lw=0.7, ls='--')
    ax.text(0.0, pooled + 0.006, f"pooled {r3(pooled)}", fontsize=6, color=fs.GREY, ha='left', va='bottom')
    ax.set_xticks(range(6)); ax.set_xticklabels(xlb, fontsize=6, rotation=35, ha='right')
    ax.set_ylim(0.70, 0.96); ax.set_xlim(-0.4, 5.4); ax.set_xlabel('Checkpoint hour since ICU admission (h)'); ax.set_ylabel('AUROC, any-AKI model')
    S = (sum(1 for i in range(6) for j in range(i + 1, 6) if MKd['strata'][j] > MKd['strata'][i])
          - sum(1 for i in range(6) for j in range(i + 1, 6) if MKd['strata'][j] < MKd['strata'][i]))  # concordant - discordant
    ax.text(0.03, 0.97, f"Exact Mann{EN}Kendall S = {S}, \u03c4 = {MKd['tau']:.3f},\nP = {MKd['exact_p']:.4f}", transform=ax.transAxes, va='top', fontsize=6)
    ax.set_title(disp + ': trend over time', fontsize=6.8, pad=3); fs.panel(ax, 'be'[r_], x=-0.42, y=1.12)
    # subgroup forest
    ax = fig.add_subplot(gs[r_, 2])
    for i, (k, lab) in enumerate(frows):
        v = SUB[k]; y = len(frows) - 1 - i
        ax.errorbar(v['auroc'], y, xerr=[[v['auroc'] - v['ci'][0]], [v['ci'][1] - v['auroc']]], fmt='D' if k == 'Overall' else mk_, color=fs.INK if k == 'Overall' else cc,
                    ms=fs.MS, capsize=1.8, elinewidth=fs.ELW, lw=0)
        ax.text(0.855, y, f"{v['auroc']:.3f} ({v['ci'][0]:.3f}{EN}{v['ci'][1]:.3f})", fontsize=6, va='center', ha='left')
        ax.text(0.995, y, f"{v['n_event_stays']}/{nfmt(v['n_stays'])}", fontsize=6, va='center', ha='left', color=fs.GREY)
    ax.text(0.855, len(frows) - 0.35, 'AUROC (95% CI)', fontsize=6, va='center', ha='left', color=fs.GREY, style='italic')
    ax.text(0.995, len(frows) - 0.35, 'Events / stays', fontsize=6, va='center', ha='left', color=fs.GREY, style='italic')
    ax.axvline(SUB['Overall']['auroc'], color=fs.GREY, lw=0.6, ls='--')
    ax.set_yticks(range(len(frows))); ax.set_yticklabels([r_[1] for r_ in frows][::-1], fontsize=6.3)
    ax.set_xlim(0.66, 1.12); ax.set_ylim(-0.6, len(frows) + 0.1); ax.set_xticks([0.7, 0.75, 0.8]); ax.spines['bottom'].set_bounds(0.66, 0.85)
    ax.set_xlabel(f'AUROC, {disp} any-AKI model', x=0.2)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.set_title(disp + ': subgroups', fontsize=6.8, pad=3); fs.panel(ax, 'cf'[r_], x=-0.32, y=1.04)
    LOG.append(f"Fig4 {pf} subgroups min/max auroc: {min(v['auroc'] for v in SUB.values())}/{max(v['auroc'] for v in SUB.values())}")
fs.save(fig, 'Fig4')

# ============================ SuppFig 1: SHAP top 15 ============================
DOMCOL = {'Creatinine kinetics': CE, 'Haemodynamics and vital signs': fs.SKY, 'Treatment exposure': fs.GREEN, 'Demographics, comorbidity and time': CJ,
          'Other laboratory and acid\u2013base': fs.GREY}
assert set(DOMCOL) == set(nr.FEATURE_DOMAIN_ORDER)
fig = plt.figure(figsize=(fs.W2, 78 * fs.MM))
ax = fig.add_axes([0.30, 0.14, 0.64, 0.78])
sh = F12['shap_top15']
for i, d in enumerate(sh):
    dom = nr.feature_domain(d['feature'])
    ax.barh(len(sh) - 1 - i, d['mean_abs'], color=DOMCOL[dom], height=0.7)
    ax.text(d['mean_abs'] + 0.02, len(sh) - 1 - i, f"{d['mean_abs']:.3f}", va='center', fontsize=6)
ax.set_yticks(range(len(sh))); ax.set_yticklabels([rlabel(d['feature']) for d in sh][::-1], fontsize=6.5)
ax.set_xlabel('Mean |SHAP value| (log-odds), any-AKI model, internal-test checkpoints'); ax.set_xlim(0, 1.7)
used = [d for d in nr.FEATURE_DOMAIN_ORDER if d in {nr.feature_domain(x_['feature']) for x_ in sh}]
ax.legend(handles=[Patch(color=DOMCOL[d], label=d) for d in used], loc='lower right', fontsize=6, borderaxespad=0.2)
fs.save(fig, 'SuppFig1')


# ============================ SuppFig 3: SHAP beeswarm + dependence (supplied matrix; layout as Yoon 2025 Figs 3-4) ============================
PRIV = sys.argv[5] if len(sys.argv) > 5 else None
assert PRIV, 'pass the private input directory (f12 SHAP matrix and hospital file; not shipped)'
sv = pd.read_parquet(os.path.join(PRIV, 'f12_shap_values.parquet')); XS = pd.read_parquet(os.path.join(PRIV, 'f12_shap_X.parquet'))
assert (sv.stay_id.values == XS.stay_id.values).all() and len(sv) == int(N['shap_n'].replace(',', ''))
sv = sv.drop(columns=['stay_id']); XS = XS.drop(columns=['stay_id']); assert list(sv.columns) == list(XS.columns) and sv.shape[1] == int(N['n_feat'])
mabs = sv.abs().mean().sort_values(ascending=False)
assert list(mabs.index[:15]) == [d['feature'] for d in F12['shap_top15']] and all(abs(mabs[d['feature']] - d['mean_abs']) < 5e-5 for d in F12['shap_top15'])
top = list(mabs.index[:20])
fig = plt.figure(figsize=(fs.W2, 110 * fs.MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.5, bottom=0.24, top=0.95, left=0.28, right=0.96)
ax = fig.add_subplot(gs[0, 0]); rng = np.random.default_rng(42); cmap = plt.get_cmap('coolwarm')
for i, f in enumerate(top[::-1]):
    v = sv[f].values; xv = XS[f].values.astype(float); fin = np.isfinite(xv)
    lo, hi = np.nanpercentile(xv, [5, 95]) if fin.any() else (0, 1)
    c = np.clip((xv[fin] - lo) / (hi - lo + 1e-12), 0, 1)
    jit = rng.uniform(-0.3, 0.3, len(v))
    ax.scatter(v[~fin], i + jit[~fin], s=1.0, color=fs.LGREY, lw=0, rasterized=True)
    ax.scatter(v[fin], i + jit[fin], s=1.0, c=cmap(c), lw=0, rasterized=True)
ax.set_yticks(range(len(top))); ax.set_yticklabels([rlabel(f) for f in top[::-1]], fontsize=6.3)
for tl, f in zip(ax.get_yticklabels(), top[::-1]):
    tl.set_color(DOMCOL[nr.feature_domain(f)])
ax.set_ylim(-0.7, len(top) - 0.3)
for i, f in enumerate(top[::-1]):
    ax.add_patch(plt.Rectangle((-0.02, i - 0.4), 0.012, 0.8, transform=ax.get_yaxis_transform(), color=DOMCOL[nr.feature_domain(f)], clip_on=False, lw=0))
ax.axvline(0, color=fs.GREY, lw=0.6); ax.set_xlabel('SHAP value (log-odds contribution)')
sm_ = plt.cm.ScalarMappable(cmap=cmap); sm_.set_array([])
cb = fig.colorbar(sm_, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1]); cb.ax.set_yticklabels(['Low', 'High'], fontsize=6)
cb.set_label('Feature value (grey: missing)', fontsize=6)
used = [d for d in nr.FEATURE_DOMAIN_ORDER if d in {nr.feature_domain(f) for f in top}]
ax.legend(handles=[Patch(color=DOMCOL[d], label=d) for d in used], loc='upper center', bbox_to_anchor=(0.45, -0.1), ncol=3, fontsize=6,
          title='Clinical domain of feature (label colour)', title_fontsize=6, handlelength=1, columnspacing=0.8)
fs.panel(ax, 'a', x=-0.42, y=1.0)
LOG.append('SuppFig4 top20: ' + ', '.join(top))
LOG.append('SuppFig4 domains in top20: ' + '; '.join(f'{d}={sum(nr.feature_domain(f) == d for f in top)}' for d in used))
LOG.append('SuppFig4 missing share in top20: ' + ', '.join(f'{f}={100 * XS[f].isna().mean():.1f}%' for f in top))
ax = fig.add_subplot(gs[0, 1])
f = 'cr_ratio_base'; assert top[0] == f
xv = XS[f].values.astype(float); v = sv[f].values; m = np.isfinite(xv)
ax.scatter(xv[m], v[m], s=2, color=fs.M4, alpha=0.3, lw=0, rasterized=True)
ax.axhline(0, color=fs.GREY, lw=0.6)
ax.set_xlim(np.nanmin(xv) - 0.03, np.nanmax(xv) + 0.03)
ax.set_xlabel('cr_ratio_base (current / baseline creatinine)', fontsize=7); ax.set_ylabel('SHAP value (log-odds contribution)')
fs.panel(ax, 'b', x=-0.25, y=1.0)
LOG.append(f'SuppFig4b cr_ratio_base: plotted {int(m.sum())} of {len(m)} checkpoints (non-missing); x range {np.nanmin(xv):.3f} to {np.nanmax(xv):.3f}')
fs.save(fig, 'SuppFig3')

# ============================ SuppFig 2: within-hospital AUROC (supplied file; layout as Cao 2026 Figs 2-3) ============================
h = pd.read_csv(os.path.join(PRIV, 'f12_hospital_aucs.csv')).sort_values('auroc', ascending=False).reset_index(drop=True)
HS = F12['hospital']
assert len(h) == HS['n_ge20'] and (h.auroc < 0.6).sum() == HS['n_below_060'] and f"{h.auroc.median():.3f}" == N['hosp_med']
assert f"{h.auroc.min():.3f}{EN}{h.auroc.max():.3f}" == N['hosp_rng'] and str((h.auroc >= 0.7).sum()) == N['hq_ge70'] and str((h.auroc >= 0.8).sum()) == N['hq_ge80']
fig = plt.figure(figsize=(fs.W15, 150 * fs.MM))
ax = fig.add_axes([0.16, 0.06, 0.78, 0.87])
y = np.arange(len(h))[::-1]
cols = [fs.ORANGE if a < 0.6 else CE for a in h.auroc]
ax.scatter(h.auroc, y, s=14, c=cols, marker='^', lw=0, zorder=3)
pooled = DSC['eICU_ge1']['auroc']
ax.axvline(HS['median'], color=fs.INK, lw=0.7, ls='--'); ax.axvline(pooled, color=fs.GREY, lw=0.7, ls=':')
ax.axvline(0.6, color=fs.ORANGE, lw=0.6, ls='-', alpha=0.6)
ax.set_yticks(y); ax.set_yticklabels([str(i + 1) for i in range(len(h))], fontsize=6)
ax.set_ylim(-1, len(h)); ax.set_xlim(0.5, 0.9); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_xlabel('Within-hospital AUROC, eICU-CRD, any-AKI model'); ax.set_ylabel('Hospital (rank by AUROC)')
ax.legend(handles=[Line2D([], [], color=fs.INK, ls='--', lw=0.7, label=f"Median {r3(HS['median'])}"),
                   Line2D([], [], color=fs.GREY, ls=':', lw=0.7, label=f"Pooled {r3(pooled)}"),
                   Line2D([], [], color=fs.ORANGE, lw=0.6, label='AUROC 0.60')], loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=6,
          borderaxespad=0.2, frameon=False)
LOG.append(f"SuppFig3: {len(h)} hospitals; median {h.auroc.median():.3f}; <0.60 {(h.auroc < 0.6).sum()}; >=0.70 {(h.auroc >= 0.7).sum()}; >=0.80 {(h.auroc >= 0.8).sum()}; pooled {pooled:.3f}")
fs.save(fig, 'SuppFig2', align=False)

open(os.path.join(OUT, 'figure_build_log.txt'), 'w').write('\n'.join(LOG) + '\n')
print('\n'.join(LOG))
