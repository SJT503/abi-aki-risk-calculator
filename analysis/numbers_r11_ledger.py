"""Round 11 (Paper 2) number ledger.
Every number used in manuscript / tables / captions / cover letter / SM is read here from data/ (f10, f11, f12, jinhu_* JSON)
plus a small set of clearly tagged companion-package cohort descriptors (data_companion/), formatted once and exposed as {N:key}.

Usage (module): N, LEDGER = load(DATA_DIR, COMPANION_DIR)
Usage (CLI):    python numbers_r11.py DATA_DIR COMPANION_DIR OUT_CSV  -> writes ledger CSV and runs all consistency / README asserts
"""
import json, os, sys, math
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
import pandas as pd

MINUS = '\u2212'
EN = '\u2013'


def r(x, nd=3):
    s = str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
    return s.replace('-', MINUS)


def sg(x, nd=3):
    return ('+' if x > 0 else '') + r(x, nd)


def n(x):
    return f'{int(x):,}'


def gfmt(x):
    return f'{x:g}'


def pfmt(p, nv):
    """P display. p==0.0 in f12 means no replicate on the opposite side of zero -> P < 2/n_valid (ceil, 4 dp)."""
    if p == 0.0:
        c = (Decimal(2) / Decimal(int(nv))).quantize(Decimal('0.0001'), rounding=ROUND_CEILING)
        return 'P < ' + format(c.normalize(), 'f')
    return f'P = {p:.3f}'


TK = {'0.05': '005', '0.08': '008', '0.1': '010', '0.15': '015', '0.2': '020'}
CK = {'int': 'int', 'M4': 'm4', 'eICU': 'ei'}   # f12 prefix -> ledger code


def load(D, DC):
    j = lambda f, d=D: json.load(open(os.path.join(d, f), encoding='utf-8'))
    F10 = j('f10_jinhua_dev_results.json'); F11 = j('f11_full_eval.json'); F12 = j('f12_full_suite.json')
    COV = j('jinhu_268_coverage.json'); T1S = j('jinhu_table1_summary.json')
    n3 = j('n3_m4_abi_grid.json', DC); n81 = j('n8_1_eicu_abi_crosswalk.json', DC); n83 = j('n8_3_eicu_abi_grid.json', DC)
    f3 = j('f3_table1.json', DC)
    N, L = {}, []

    def add(key, val, src, field):
        assert key not in N, key
        N[key] = val; L.append({'key': key, 'value': val, 'source_file': src, 'field': field})

    DSC, PAR, CAL, DEP = F12['discrimination'], F12['paired'], F12['calibration'], F12['deployment']
    T1 = F12['table1']

    # ---------- discrimination (3 cohorts x 3 tiers) ----------
    for g in (1, 2, 3):
        for pf, c in CK.items():
            d = DSC[f'{pf}_ge{g}']; s = 'f12_full_suite.json'
            add(f'auc{g}_{c}', r(d['auroc']), s, f'discrimination.{pf}_ge{g}.auroc')
            add(f'ci{g}_{c}', f"{r(d['ci'][0])}{EN}{r(d['ci'][1])}", s, f'discrimination.{pf}_ge{g}.ci')
            add(f'auprc{g}_{c}', f"{d['auprc']:.4f}", s, f'discrimination.{pf}_ge{g}.auprc')
            add(f'pos{g}_{c}', n(d['pos_ckpt']), s, f'discrimination.{pf}_ge{g}.pos_ckpt')
            add(f'evs{g}_{c}', n(d['event_stays']), s, f'discrimination.{pf}_ge{g}.event_stays')
    # internal Tier 2 is underpowered: the CI upper bound is 1.0 (degenerate)
    assert DSC['int_ge3']['event_stays'] == 3 and DSC['int_ge3']['pos_ckpt'] == 15 and DSC['int_ge3']['ci'][1] == 1.0
    add('int_t2_ci_hi', r(DSC['int_ge3']['ci'][1]), 'f12_full_suite.json', 'discrimination.int_ge3.ci[1] (degenerate upper bound, 3 event stays)')

    # ---------- paired comparisons ----------
    for pf, c in CK.items():
        for k, key in (('ge2_minus_ge1', '21'), ('ge3_minus_ge1', '31'), ('ge3_minus_ge2', '32')):
            v = PAR[f'{pf}_{k}']; s = 'f12_full_suite.json'; f = f'paired.{pf}_{k}'
            add(f'pr{key}_{c}', sg(v['delta']), s, f + '.delta')
            add(f'pr{key}_{c}_ci', f"{r(v['ci'][0])} to {r(v['ci'][1])}", s, f + '.ci')
            add(f'pr{key}_{c}_p', pfmt(v['p'], v['n_valid']), s, f + '.p' + (' (0.0 -> P < 2/n_valid)' if v['p'] == 0.0 else ''))
            add(f'pr{key}_{c}_nv', n(v['n_valid']), s, f + '.n_valid')
            add(f'pr{key}_{c}_pnum', f"{v['p']:.3f}", s, f + '.p (3-dp as stored)')
    for c in CK.values():  # display self-consistency: sign of delta equals sign of the AUROC point difference
        for a, b, key in ((2, 1, '21'), (3, 1, '31'), (3, 2, '32')):
            pd_ = float(N[f'auc{a}_{c}']) - float(N[f'auc{b}_{c}'])
            assert (pd_ > 0) == (not N[f'pr{key}_{c}'].startswith(MINUS)), (c, key)

    # ---------- LR baseline (f12 key 'delong_ext' is a mis-named paired stay bootstrap; delta = LR - model) ----------
    DL = F12['delong_ext']
    assert DL['delta'] < 0 and DL['ci'][1] < 0 and DL['p'] == 0.0
    add('lr_m4_diff', sg(-DL['delta']), 'f12_full_suite.json', 'delong_ext.delta (stored as LR - model; sign flipped = model - LR)')
    add('lr_m4_ci', f"{r(-DL['ci'][1])} to {r(-DL['ci'][0])}", 'f12_full_suite.json', 'delong_ext.ci (negated, bounds swapped)')
    add('lr_m4_p', pfmt(DL['p'], DL['n_valid']), 'f12_full_suite.json', 'delong_ext.p (0.0 -> P < 2/n_valid)')
    add('lr_m4_nv', n(DL['n_valid']), 'f12_full_suite.json', 'delong_ext.n_valid')

    # ---------- severity gradient ----------
    for pf, c in (('M4', 'm4'), ('eICU', 'ei')):
        for k, key in (('any', 'any'), ('s12', 's12'), ('s3', 's3')):
            v = DSC[f'{pf}_grad_{k}']; s = 'f12_full_suite.json'
            add(f'sev_{key}_{c}', r(v['auroc']), s, f'discrimination.{pf}_grad_{k}.auroc')
            add(f'sev_{key}_{c}_ci', f"{r(v['ci'][0])}{EN}{r(v['ci'][1])}", s, f'discrimination.{pf}_grad_{k}.ci')
            add(f'sev_{key}_{c}_pos', n(v['n_pos']), s, f'discrimination.{pf}_grad_{k}.n_pos')
            add(f'sev_{key}_{c}_stays', n(v['n_stays']), s, f'discrimination.{pf}_grad_{k}.n_stays')
        a, b, s3 = DSC[f'{pf}_grad_any'], DSC[f'{pf}_grad_s12'], DSC[f'{pf}_grad_s3']
        assert b['n_pos'] + s3['n_pos'] == a['n_pos'] and b['n_stays'] + s3['n_stays'] == a['n_stays'], pf
        assert a['n_pos'] == DSC[f'{pf}_ge1']['pos_ckpt'] and a['n_stays'] == DSC[f'{pf}_ge1']['event_stays']
        assert s3['n_pos'] == DSC[f'{pf}_ge3']['pos_ckpt'] and s3['n_stays'] == DSC[f'{pf}_ge3']['event_stays']
        assert a['auroc'] == DSC[f'{pf}_ge1']['auroc']

    # ---------- calibration (Tier 1 only) ----------
    add('recal_a', f"{CAL['a']:.6f}", 'f12_full_suite.json', 'calibration.a')
    add('recal_b', f"{CAL['b']:.6f}", 'f12_full_suite.json', 'calibration.b')
    assert (F10['calibration_ge1']['a'], F10['calibration_ge1']['b']) == (CAL['a'], CAL['b'])
    for pf, c in CK.items():
        for st in ('raw', 'recal'):
            add(f'cal_{c}_{st}_slope', f"{CAL[pf][st]['slope']:.3f}", 'f12_full_suite.json', f'calibration.{pf}.{st}.slope')
            add(f'cal_{c}_{st}_ece', f"{CAL[pf][st]['ece']:.4f}", 'f12_full_suite.json', f'calibration.{pf}.{st}.ece')
    # f10 / f11 overlaps
    assert F10['calibration_ge1']['test_raw'] == CAL['int']['raw'] and F10['calibration_ge1']['test_recal'] == CAL['int']['recal']
    assert F11['M4_ext']['calib_ge1_frozen_layer']['slope'] == CAL['M4']['recal']['slope']
    assert F11['eICU_ext']['calib_ge1_frozen_layer']['slope'] == CAL['eICU']['recal']['slope']
    for g, k in ((1, 'ge1'), (2, 'ge2'), (3, 'ge3')):
        r10 = F10['results'][k]; d12 = DSC[f'int_ge{g}']
        assert (r10['internal_test_auroc'], r10['ci'], r10['auprc'], r10['test_pos_ckpt'], r10['test_event_stays']) == \
               (d12['auroc'], d12['ci'], d12['auprc'], d12['pos_ckpt'], d12['event_stays']), g
        for pf in ('M4', 'eICU'):
            e11 = F11[f'{pf}_ext'][k]; d = DSC[f'{pf}_ge{g}']
            assert (e11['auroc'], e11['ci'], e11['auprc'], e11['pos_ckpt'], e11['event_stays']) == \
                   (d['auroc'], d['ci'], d['auprc'], d['pos_ckpt'], d['event_stays']), (pf, g)
    assert F11['internal_ge1'] == {'auroc': DSC['int_ge1']['auroc'], 'ci': DSC['int_ge1']['ci'], 'auprc': DSC['int_ge1']['auprc']}

    # ---------- deployment v2 ----------
    STAYS = {'int': T1['Jinhua_test']['stays'], 'M4': T1['M4_ext']['stays'], 'eICU': T1['eICU_ext']['stays']}
    for pf, c in CK.items():
        for t, tk in TK.items():
            v = DEP[pf][f'thr{t}']; s = 'f12_full_suite.json'; f = f'deployment.{pf}.thr{t}'; p = f'dp_{c}_{tk}_'
            assert abs(v['capture'] - v['n_captured'] / v['n_event_stays']) <= 0.0006, (pf, t)
            assert abs(v['NNE'] - v['n_alert_stays'] / v['n_captured']) <= 0.005, (pf, t)
            assert abs(v['pct_stays_alerted'] - 100 * v['n_alert_stays'] / STAYS[pf]) <= 0.05, (pf, t)
            assert v['n_event_stays'] == DSC[f'{pf}_ge1']['event_stays']
            add(p + 'cap', r(100 * v['capture'], 1), s, f + '.capture x 100')
            add(p + 'lead', gfmt(v['lead_median_h']), s, f + '.lead_median_h')
            add(p + 'iqr', f"{gfmt(v['lead_iqr'][0])}{EN}{gfmt(v['lead_iqr'][1])}", s, f + '.lead_iqr')
            add(p + 'nne', f"{v['NNE']:.2f}", s, f + '.NNE')
            add(p + 'fa', f"{v['FA_per_100ptd']:.2f}", s, f + '.FA_per_100ptd')
            add(p + 'pct', f"{v['pct_stays_alerted']:.1f}", s, f + '.pct_stays_alerted')
            add(p + 'ncap', n(v['n_captured']), s, f + '.n_captured')
            add(p + 'nev', n(v['n_event_stays']), s, f + '.n_event_stays')
            add(p + 'nal', n(v['n_alert_stays']), s, f + '.n_alert_stays')
    # monotonic sanity: capture and alerted fraction fall with the threshold in every cohort
    for pf in CK:
        caps = [DEP[pf][f'thr{t}']['capture'] for t in TK]; pcts = [DEP[pf][f'thr{t}']['pct_stays_alerted'] for t in TK]
        assert caps == sorted(caps, reverse=True) and pcts == sorted(pcts, reverse=True), pf
    # ranges across the 5 thresholds (derived)
    for pf, c in CK.items():
        caps = [DEP[pf][f'thr{t}']['capture'] for t in TK]; nnes = [DEP[pf][f'thr{t}']['NNE'] for t in TK]
        add(f'dp_{c}_cap_hi', r(100 * max(caps), 1), 'f12_full_suite.json', f'max over deployment.{pf}.thr*.capture x 100')
        add(f'dp_{c}_cap_lo', r(100 * min(caps), 1), 'f12_full_suite.json', f'min over deployment.{pf}.thr*.capture x 100')
        add(f'dp_{c}_nne_hi', f'{max(nnes):.2f}', 'f12_full_suite.json', f'max over deployment.{pf}.thr*.NNE')
        add(f'dp_{c}_nne_lo', f'{min(nnes):.2f}', 'f12_full_suite.json', f'min over deployment.{pf}.thr*.NNE')
    add('dp_n_thr', str(len(TK)), 'f12_full_suite.json', 'number of deployment thresholds')
    # NNE equality across cohorts at thr 0.15 (M4 1531/239 and eICU 1666/260) is a numerical coincidence; show the divisions
    add('dp_m4_015_nne_calc', f"{DEP['M4']['thr0.15']['n_alert_stays'] / DEP['M4']['thr0.15']['n_captured']:.3f}", 'f12_full_suite.json', 'n_alert_stays / n_captured (M4 thr0.15)')
    add('dp_ei_015_nne_calc', f"{DEP['eICU']['thr0.15']['n_alert_stays'] / DEP['eICU']['thr0.15']['n_captured']:.3f}", 'f12_full_suite.json', 'n_alert_stays / n_captured (eICU thr0.15)')

    # ---------- hospitals (eICU) ----------
    HS = F12['hospital']
    add('hosp_n', str(HS['n_ge20']), 'f12_full_suite.json', 'hospital.n_ge20')
    add('hosp_med', r(HS['median']), 'f12_full_suite.json', 'hospital.median')
    add('hosp_rng', f"{r(HS['min'])}{EN}{r(HS['max'])}", 'f12_full_suite.json', 'hospital.min/max')
    add('hosp_lt60', str(HS['n_below_060']), 'f12_full_suite.json', 'hospital.n_below_060')
    add('hosp_ge70_note', 'not computed', 'f12_full_suite.json', 'per-hospital values not in data/ (f12_hospital_aucs.csv pending)')
    hm = pd.read_parquet(os.path.join(DC, 'f9_hospital_map.parquet'))
    ids = pd.read_parquet(os.path.join(DC, 'eicu_grid_stay_ids.parquet')).stay_id.unique()
    nh = hm[hm.stay_id.isin(ids)].hospitalid.nunique()
    assert nh == 171 and hm.stay_id.isin(ids).sum() == len(ids) == T1['eICU_ext']['stays'] == 9071
    add('ei_hosp', str(nh), 'data_companion/f9_hospital_map.parquet + eicu_grid_stay_ids.parquet (same cohort, companion package)',
        'hospitalid.nunique() over the 9,071 eICU grid stays')

    # ---------- Mann-Kendall (eICU, 6 checkpoint-hour strata) ----------
    MK = F12['mk']; S = len(MK['strata']); pairs = S * (S - 1) // 2
    assert S == 6 and math.factorial(S) == 720 and MK['tau'] == 1.0
    assert MK['strata'] == sorted(MK['strata'])  # strictly increasing: all 15 pairs concordant
    assert abs(MK['exact_p'] - round(2 * 1 / 720, 4)) < 1e-12  # only the identity ordering reaches S=15
    add('mk_S', str(pairs), 'f12_full_suite.json', 'derived: all 15 pairs concordant (strata strictly increasing, tau = 1.0)')
    add('mk_tau', f"{MK['tau']:.3f}", 'f12_full_suite.json', 'mk.tau')
    add('mk_p', f"P = {MK['exact_p']:.4f}", 'f12_full_suite.json', 'mk.exact_p (= 2 x 1/720)')
    add('mk_p_asym', f"P = {MK['p_asym']:.3f}", 'f12_full_suite.json', 'mk.p_asym')
    for i, v in enumerate(MK['strata']):
        add(f'mk_s{i}', r(v), 'f12_full_suite.json', f'mk.strata[{i}]')

    # ---------- subgroups (eICU only) ----------
    SGK = {'Overall': 'overall', 'Age <65': 'age_lt65', 'Age >=65': 'age_ge65', 'Male': 'male', 'Female': 'female', 'TBI': 'tbi',
           'SAH/ICH': 'sahich', 'IS/unspec': 'isunspec', 'CKD': 'ckd', 'No CKD': 'nockd'}
    SUB = F12['subgroups']; assert list(SUB) == list(SGK)
    for k, key in SGK.items():
        v = SUB[k]
        add(f'sg_{key}', f"{v['auroc']:.3f} ({v['ci'][0]:.3f}{EN}{v['ci'][1]:.3f})", 'f12_full_suite.json', f'subgroups[{k}].auroc/ci')
        add(f'sg_{key}_auc', f"{v['auroc']:.3f}", 'f12_full_suite.json', f'subgroups[{k}].auroc')
        add(f'sg_{key}_n', f"{n(v['n_stays'])} / {n(v['n_event_stays'])}", 'f12_full_suite.json', f'subgroups[{k}].n_stays/n_event_stays')
    assert SUB['Overall']['n_stays'] == STAYS['eICU'] and SUB['Overall']['n_event_stays'] == DSC['eICU_ge1']['event_stays']
    assert SUB['Overall']['ci'] == DSC['eICU_ge1']['ci'] and SUB['Overall']['auroc'] == round(DSC['eICU_ge1']['auroc'], 3)
    for a, b in (('Age <65', 'Age >=65'), ('Male', 'Female'), ('CKD', 'No CKD')):
        assert SUB[a]['n_stays'] + SUB[b]['n_stays'] == SUB['Overall']['n_stays'] and SUB[a]['n_event_stays'] + SUB[b]['n_event_stays'] == SUB['Overall']['n_event_stays']
    assert SUB['TBI']['n_stays'] + SUB['SAH/ICH']['n_stays'] + SUB['IS/unspec']['n_stays'] <= SUB['Overall']['n_stays']
    nsub = len(SUB) - 1
    add('sg_count', str(nsub), 'f12_full_suite.json', 'len(subgroups) - Overall')
    add('sg_min', f"{min(v['auroc'] for k, v in SUB.items() if k != 'Overall'):.3f}", 'f12_full_suite.json', 'min subgroup auroc (excl. Overall)')
    add('sg_other_stays', n(SUB['Overall']['n_stays'] - SUB['TBI']['n_stays'] - SUB['SAH/ICH']['n_stays'] - SUB['IS/unspec']['n_stays']),
        'f12_full_suite.json', 'derived: Overall - TBI - SAH/ICH - IS/unspec stays (anoxic, encephalitis, other)')
    add('sg_tbi_pct', f"{100 * SUB['TBI']['n_stays'] / SUB['Overall']['n_stays']:.1f}", 'f12_full_suite.json', 'derived: subgroups[TBI].n_stays / Overall.n_stays x 100')

    # ---------- f27: M4 MK + subgroups (closes the Fig4 eICU-only asymmetry; PI order 2026-10-05) ----------
    F27 = j('f27_m4_subgroups_mk.json'); s27 = 'f27_m4_subgroups_mk.json'
    M4S = F27['subgroups']; assert list(M4S) == list(SUB)
    assert M4S['Overall']['n_stays'] == STAYS['M4'] and M4S['Overall']['n_event_stays'] == DSC['M4_ge1']['event_stays']
    assert M4S['Overall']['ci'] == DSC['M4_ge1']['ci'] and M4S['Overall']['auroc'] == round(DSC['M4_ge1']['auroc'], 3)
    for a, b in (('Age <65', 'Age >=65'), ('Male', 'Female'), ('CKD', 'No CKD')):
        assert M4S[a]['n_stays'] + M4S[b]['n_stays'] == M4S['Overall']['n_stays'] and M4S[a]['n_event_stays'] + M4S[b]['n_event_stays'] == M4S['Overall']['n_event_stays']
    MK4 = F27['mk']; assert len(MK4['strata']) == 6
    assert 'S' in MK4  # f27 >= 2026-10-06 stores the proper Mann-Kendall S (concordant - discordant)
    assert MK4['S'] == MK4['n_concordant'] - MK4['n_discordant']
    assert abs((MK4['S'] / 15) - MK4['tau']) < 0.002  # S and tau must agree
    add('mk_m4_S', str(MK4['S']), s27, 'mk.S (concordant minus discordant; = 7 for the M4 strata)')
    add('mk_m4_tau', f"{MK4['tau']:.3f}", s27, 'mk.tau')
    add('mk_m4_p', f"P = {MK4['exact_p']:.4f}", s27, 'mk.exact_p')
    add('mk_m4_p_asym', f"P = {MK4['p_asym']:.3f}", s27, 'mk.p_asym')
    for i, v in enumerate(MK4['strata']):
        add(f'mk_m4_s{i}', r(v), s27, f'mk.strata[{i}]')
    for k, key in SGK.items():
        v = M4S[k]
        add(f'sg_m4_{key}', f"{v['auroc']:.3f} ({v['ci'][0]:.3f}{EN}{v['ci'][1]:.3f})", s27, f'subgroups[{k}].auroc/ci')
        add(f'sg_m4_{key}_auc', f"{v['auroc']:.3f}", s27, f'subgroups[{k}].auroc')
        add(f'sg_m4_{key}_n', f"{n(v['n_stays'])} / {n(v['n_event_stays'])}", s27, f'subgroups[{k}].n_stays/n_event_stays')
    add('sg_m4_min', f"{min(v['auroc'] for k, v in M4S.items() if k != 'Overall'):.3f}", s27, 'min M4 subgroup auroc (excl. Overall)')
    add('sg_m4_other_stays', n(M4S['Overall']['n_stays'] - M4S['TBI']['n_stays'] - M4S['SAH/ICH']['n_stays'] - M4S['IS/unspec']['n_stays']),
        s27, 'derived: Overall - TBI - SAH/ICH - IS/unspec stays (anoxic, encephalitis, other)')
    add('sg_m4_tbi_pct', f"{100 * M4S['TBI']['n_stays'] / M4S['Overall']['n_stays']:.1f}", s27, 'derived: subgroups[TBI].n_stays / Overall.n_stays x 100')

    # ---------- SHAP (f12 authoritative), selection, EPPP ----------
    for i, d in enumerate(F12['shap_top15']):
        add(f'shap{i + 1}', d['feature'], 'f12_full_suite.json', f'shap_top15[{i}].feature')
        add(f'shapv{i + 1}', f"{d['mean_abs']:.3f}", 'f12_full_suite.json', f'shap_top15[{i}].mean_abs')
    assert len(F12['shap_top15']) == 15
    s11 = [d['feature'] for d in F11['shap_top15']]; s12 = [d['feature'] for d in F12['shap_top15']]
    add('shap_f11_overlap', str(len(set(s11) & set(s12))), 'f11_full_eval.json vs f12_full_suite.json', 'size of intersection of the two top-15 SHAP lists')
    add('shap_f11_top5_same', str(s11[:5] == s12[:5]), 'f11_full_eval.json vs f12_full_suite.json', 'top-5 identical in order (f11 vs f12)')
    add('shap_n', n(F10['split']['test_ckpt']), 'f12_full_suite.py (size = min(4000, te.sum())) and f10 split.test_ckpt', 'derived: all internal-test checkpoints')
    assert F10['split']['test_ckpt'] < 4000
    SEL = F12['selection']
    assert SEL['pool'] == F10['candidate_pool'] == 280 and SEL['selected'] == F10['n_selected'] == F11['selected_n'] == F12['eppp']['params'] == 166
    assert SEL['curve'] == F10['curve_n_features']
    add('n_cand', str(SEL['pool']), 'f12_full_suite.json', 'selection.pool'); add('n_feat', str(SEL['selected']), 'f12_full_suite.json', 'selection.selected')
    add('n_nan_drop', str(F10['all_nan_dropped']), 'f10_jinhua_dev_results.json', 'all_nan_dropped')
    for k, v in SEL['curve'].items():
        add(f"curve_{k.replace('.0%', '').replace('%', '')}", str(v), 'f12_full_suite.json', f'selection.curve[{k}]')
    for i, f in enumerate(F11['selected_features_top20']):
        add(f'cvgain{i + 1}', f, 'f11_full_eval.json', f'selected_features_top20[{i}]')
    assert F11['missing_in_m4'] == [] and F11['missing_in_eicu'] == []
    add('n_missing_ext', '0', 'f11_full_eval.json', 'len(missing_in_m4) + len(missing_in_eicu)')
    E = F12['eppp']
    ev_ck, ev_pt = F10['train_events_ge1_ckpt'], F10['train_event_subjects_ge1']
    assert ev_ck == T1['Jinhua_dev']['ge1_ckpt'] == 819 and ev_pt == T1['Jinhua_dev']['ge1_stays'] == 155
    assert round(ev_ck / E['params'], 1) == E['ckpt_level'] and round(ev_pt / E['params'], 2) == E['patient_level']
    assert F10['eppp'] == {'params': E['params'], 'ckpt_level_dev': E['ckpt_level'], 'patient_level_dev': E['patient_level']}
    add('eppp_params', str(E['params']), 'f12_full_suite.json', 'eppp.params')
    add('eppp_ckpt', f"{E['ckpt_level']:.1f}", 'f12_full_suite.json', 'eppp.ckpt_level (= 819/166 checked)')
    add('eppp_pat', f"{E['patient_level']:.2f}", 'f12_full_suite.json', 'eppp.patient_level (= 155/166 checked)')
    add('eppp_ev_ckpt', n(ev_ck), 'f10_jinhua_dev_results.json', 'train_events_ge1_ckpt')
    add('eppp_ev_pat', n(ev_pt), 'f10_jinhua_dev_results.json', 'train_event_subjects_ge1')
    add('eppp_t2_ev', n(T1['Jinhua_dev']['ge3_ckpt']), 'f12_full_suite.json', 'table1.Jinhua_dev.ge3_ckpt (derived training Tier-2 events)')
    add('eppp_t2', f"{T1['Jinhua_dev']['ge3_ckpt'] / E['params']:.2f}", 'f12_full_suite.json', 'derived: table1.Jinhua_dev.ge3_ckpt / eppp.params (68/166)')

    # ---------- SM5 arms ----------
    S5 = F12['sm5']
    for g in (1, 2, 3):
        for pf, c in CK.items():
            add(f'fill0_{g}_{c}', r(S5['fill0'][f'ge{g}'][pf]), 'f12_full_suite.json', f'sm5.fill0.ge{g}.{pf}')
        for pf in ('M4', 'eICU'):
            add(f'ts_{g}_{CK[pf]}', r(S5['threshold_specific'][f'ge{g}'][pf]), 'f12_full_suite.json', f'sm5.threshold_specific.ge{g}.{pf}')

    # ---------- DCA / calibration points (f11; recalibrated Tier-1 probabilities) ----------
    dca = {'int': F11['internal_dca'], 'm4': F11['M4_ext']['dca_ge1'], 'ei': F11['eICU_ext']['dca_ge1']}
    for c, rows in dca.items():
        dd = pd.DataFrame(rows)
        assert list(dd.thr) == [round(0.05 * i, 2) for i in range(1, 11)]
        assert (dd.nb_model > dd.nb_treat_all).all()                       # always above treat-all
        pos = dd[dd.nb_model > 0]
        assert list(pos.thr) == sorted(pos.thr) and (pos.thr.max() - pos.thr.min()) / 0.05 + 1 == len(pos)  # contiguous from the low end
        add(f'dca_{c}_pos_hi', f'{pos.thr.max():.2f}', 'f11_full_eval.json', 'max threshold with nb_model > 0')
        add(f'dca_{c}_n_pos', str(len(pos)), 'f11_full_eval.json', 'number of the 10 thresholds with nb_model > 0')
        add(f'dca_{c}_nb05', f"{dd.nb_model.iloc[0]:.4f}", 'f11_full_eval.json', 'nb_model at threshold 0.05')
        add(f'dca_{c}_ta05', f"{r(dd.nb_treat_all.iloc[0], 4)}", 'f11_full_eval.json', 'nb_treat_all at threshold 0.05')
    assert dca['int'] and len(pd.DataFrame(dca['int'])[lambda d: d.nb_model > 0]) == 10
    add('dca_lo', '0.05', 'f11_full_eval.json', 'dca thr min'); add('dca_hi', '0.50', 'f11_full_eval.json', 'dca thr max')
    for c, cc in (('int', F11['internal_cal_curve']), ('m4', F11['M4_ext']['cal_curve_ge1']), ('ei', F11['eICU_ext']['cal_curve_ge1'])):
        assert len(cc) == 2 and len(cc[0]) == len(cc[1]) == 10

    # ---------- cohort tables (Table 1) ----------
    cohort = {'dev': T1['Jinhua_dev'], 'test': T1['Jinhua_test'], 'm4': T1['M4_ext'], 'ei': T1['eICU_ext']}
    DASH = '\u2014'
    s_ = 'f12_full_suite.json'
    for k, row in cohort.items():
        add(f't1_{k}_stays', n(row['stays']), s_, f'table1.{k}.stays')
        add(f't1_{k}_ckpt', n(row['ckpt']), s_, f'table1.{k}.ckpt')
        add(f't1_{k}_ge1_ckpt', n(row['ge1_ckpt']), s_, f'table1.{k}.ge1_ckpt')
        add(f't1_{k}_ge1_stays', n(row['ge1_stays']), s_, f'table1.{k}.ge1_stays')
        add(f't1_{k}_ge2_ckpt', n(row['ge2_ckpt']) if 'ge2_ckpt' in row else DASH, s_, f'table1.{k}.ge2_ckpt (dash = not recorded)')
        add(f't1_{k}_ge3_ckpt', n(row['ge3_ckpt']), s_, f'table1.{k}.ge3_ckpt')
        add(f't1_{k}_ge3_stays', n(row['ge3_stays']) if 'ge3_stays' in row else DASH, s_, f'table1.{k}.ge3_stays (dash = not recorded)')
        add(f't1_{k}_rate1', f"{100 * row['ge1_ckpt'] / row['ckpt']:.1f}", s_, 'derived: ge1_ckpt / ckpt x 100')
        add(f't1_{k}_stayrate1', f"{100 * row['ge1_stays'] / row['stays']:.1f}", s_, 'derived: ge1_stays / stays x 100')
        add(f't1_{k}_perstay', f"{row['ckpt'] / row['stays']:.1f}", s_, 'derived: ckpt / stays')
    # ge2 event stays: only recorded in the discrimination block (internal test / M4 / eICU)
    add('t1_dev_ge2_stays', DASH, s_, 'not recorded for the development split')
    add('t1_test_ge2_stays', n(DSC['int_ge2']['event_stays']), s_, 'discrimination.int_ge2.event_stays')
    add('t1_m4_ge2_stays', n(DSC['M4_ge2']['event_stays']), s_, 'discrimination.M4_ge2.event_stays')
    add('t1_ei_ge2_stays', n(DSC['eICU_ge2']['event_stays']), s_, 'discrimination.eICU_ge2.event_stays')
    # cross-check table1 vs discrimination (ge1 / ge2 / ge3 checkpoints and stays)
    for pf, k in (('int', 'test'), ('M4', 'm4'), ('eICU', 'ei')):
        row = cohort[k]
        assert row['ge1_ckpt'] == DSC[f'{pf}_ge1']['pos_ckpt'] and row['ge1_stays'] == DSC[f'{pf}_ge1']['event_stays']
        assert row['ge3_ckpt'] == DSC[f'{pf}_ge3']['pos_ckpt'] and row['ge3_stays'] == DSC[f'{pf}_ge3']['event_stays']
        if 'ge2_ckpt' in row:
            assert row['ge2_ckpt'] == DSC[f'{pf}_ge2']['pos_ckpt']
    # test-split ge2/ge3 checkpoints present only through discrimination
    add('t1_test_ge2_ckpt_disc', n(DSC['int_ge2']['pos_ckpt']), s_, 'discrimination.int_ge2.pos_ckpt (table1.Jinhua_test has no ge2_ckpt)')
    # Jinhua descriptors
    for k, row in (('dev', T1['Jinhua_dev']), ('test', T1['Jinhua_test'])):
        a = row['age']
        add(f't1_{k}_age', f'{a[0]:.0f} ({a[1]:.0f}{EN}{a[2]:.0f})', s_, f'table1.{k}.age')
        add(f't1_{k}_male', f"{row['male_pct']:.1f}", s_, f'table1.{k}.male_pct')
        add(f't1_{k}_subj', n(row['subjects']), s_, f'table1.{k}.subjects')
    add('t1_dev_charlson', f"{T1['Jinhua_dev']['charlson']:.0f} (IQR not recorded)", s_, 'table1.Jinhua_dev.charlson (median only)')
    add('t1_test_charlson', 'not computed', s_, 'table1.Jinhua_test has no charlson field')
    add('t1_dev_crbase', 'not computed', s_, 'not in data/'); add('t1_test_crbase', 'not computed', s_, 'not in data/')
    # funnel
    SP = F10['split']
    assert SP['train_subj'] + SP['test_subj'] == F10['subjects'] == 1031
    assert T1['Jinhua_dev']['subjects'] == SP['train_subj'] and T1['Jinhua_test']['subjects'] == SP['test_subj']
    assert T1['Jinhua_dev']['stays'] + T1['Jinhua_test']['stays'] == F10['stays'] == T1S['n_at_risk_stays'] == 1033
    assert T1['Jinhua_dev']['ckpt'] == SP['train_ckpt'] and T1['Jinhua_test']['ckpt'] == SP['test_ckpt']
    assert SP['train_ckpt'] + SP['test_ckpt'] == T1S['n_checkpoints'] == 12483
    add('jh_stays', n(F10['stays']), 'f10_jinhua_dev_results.json', 'stays (= jinhu_table1_summary.n_at_risk_stays)')
    add('jh_subjects', n(F10['subjects']), 'f10_jinhua_dev_results.json', 'subjects')
    add('jh_ckpt', n(T1S['n_checkpoints']), 'jinhu_table1_summary.json', 'n_checkpoints (= 9,837 + 2,646)')
    add('jh_train_ckpt', n(SP['train_ckpt']), 'f10_jinhua_dev_results.json', 'split.train_ckpt'); add('jh_test_ckpt', n(SP['test_ckpt']), 'f10_jinhua_dev_results.json', 'split.test_ckpt')
    add('jh_train_subj', n(SP['train_subj']), 'f10_jinhua_dev_results.json', 'split.train_subj'); add('jh_test_subj', n(SP['test_subj']), 'f10_jinhua_dev_results.json', 'split.test_subj')
    a = T1S['age_median_iqr']; c_ = T1S['charlson_median_iqr']; b = T1S['cr_base_median_iqr']
    add('jh_age', f'{a[0]:.0f} ({a[1]:.0f}{EN}{a[2]:.0f})', 'jinhu_table1_summary.json', 'age_median_iqr (pooled 1,033 stays)')
    add('jh_male', f"{T1S['male_pct']:.1f}", 'jinhu_table1_summary.json', 'male_pct (pooled)')
    add('jh_charlson', f'{c_[0]:.0f} ({c_[1]:.0f}{EN}{c_[2]:.0f})', 'jinhu_table1_summary.json', 'charlson_median_iqr (pooled)')
    add('jh_crbase', f'{b[0]:.2f} ({b[1]:.2f}{EN}{b[2]:.2f})', 'jinhu_table1_summary.json', 'cr_base_median_iqr (pooled, mg/dL after conversion)')
    add('jh_los', f"{T1S['los_icu_h_median']:.1f}", 'jinhu_table1_summary.json', 'los_icu_h_median (pooled)')
    add('jh_tbi', f"{T1S['subtype_tbi_pct']:.1f}", 'jinhu_table1_summary.json', 'subtype_tbi_pct (pooled)')
    wm = (T1['Jinhua_dev']['male_pct'] * T1['Jinhua_dev']['stays'] + T1['Jinhua_test']['male_pct'] * T1['Jinhua_test']['stays']) / 1033
    add('jh_male_weighted_check', f'{wm:.2f}', 'f12_full_suite.json table1 (derived)', 'stay-weighted mean of dev/test male_pct (consistency check only; reported pooled value is jh_male)')
    assert abs(wm - T1S['male_pct']) < 0.2

    # ---------- companion-package descriptors of the SAME external cohorts (no model performance) ----------
    cs = 'data_companion/'
    assert n3['grid']['patients'] == T1['M4_ext']['stays'] == 7442 and n3['grid']['checkpoints'] == T1['M4_ext']['ckpt'] == 75638
    assert n83['grid']['patients'] == T1['eICU_ext']['stays'] == 9071 and n83['grid']['checkpoints'] == T1['eICU_ext']['ckpt'] == 84866
    add('m4_stays', n(n3['n_stays']), cs + 'n3_m4_abi_grid.json (same cohort, companion package)', 'n_stays')
    add('m4_atrisk', n(n3['at_risk']), cs + 'n3_m4_abi_grid.json (same cohort, companion package)', 'at_risk')
    add('ei_stays_cw', n(n81['n_stays']), cs + 'n8_1_eicu_abi_crosswalk.json (same cohort, companion package)', 'n_stays')
    add('ei_hosp_all', n(n81['n_hospitals']), cs + 'n8_1_eicu_abi_crosswalk.json (same cohort, companion package)', 'n_hospitals')
    add('ei_atrisk', n(n83['at_risk']), cs + 'n8_3_eicu_abi_grid.json (same cohort, companion package)', 'at_risk')
    add('ei_uo_excl', n(n83['excl_uo_prev24']), cs + 'n8_3_eicu_abi_grid.json (same cohort, companion package)', 'excl_uo_prev24')
    er = f3['rows'][-1]
    assert er['cohort'] == 'eICU external' and er['stays'] == 9071 and er['ge1_event_stays'] == T1['eICU_ext']['ge1_stays'] and er['ge3_event_stays'] == T1['eICU_ext']['ge3_stays']
    assert er['ge1_pos_ckpt'] == T1['eICU_ext']['ge1_ckpt'] and er['ge2_pos_ckpt'] == T1['eICU_ext']['ge2_ckpt'] and er['ge3_pos_ckpt'] == T1['eICU_ext']['ge3_ckpt']
    fs_ = cs + 'f3_table1.json eICU row (same cohort, companion package)'
    a = er['age_median_iqr']; c_ = er['charlson_median_iqr']; b = er['cr_base_median_iqr']
    add('t1_ei_age', f'{a[0]:.0f} ({a[1]:.0f}{EN}{a[2]:.0f})', fs_, 'age_median_iqr')
    add('t1_ei_male', f"{er['male_pct']:.1f}", fs_, 'male_pct')
    add('t1_ei_charlson', f'{c_[0]:.0f} ({c_[1]:.0f}{EN}{c_[2]:.0f})', fs_, 'charlson_median_iqr')
    add('t1_ei_crbase', f'{b[0]:.2f} ({b[1]:.2f}{EN}{b[2]:.2f})', fs_, 'cr_base_median_iqr')
    for k in ('age', 'male', 'charlson', 'crbase'):
        add(f't1_m4_{k}', 'not computed', 'n/a', 'M4 descriptors cannot be pooled from the companion train/test strata without estimation')
    mix = n81['subtype_mix_audit']
    for k in mix['m4_pct']:
        add(f'mix_m4_{k}', f"{mix['m4_pct'][k]:.1f}", cs + 'n8_1_eicu_abi_crosswalk.json (same cohort, companion package)', f'subtype_mix_audit.m4_pct.{k}')
        add(f'mix_ei_{k}', f"{mix['eicu_pct'][k]:.1f}", cs + 'n8_1_eicu_abi_crosswalk.json (same cohort, companion package)', f'subtype_mix_audit.eicu_pct.{k}')

    # ---------- coverage of the companion 268-feature set (SM2/SM3 only) ----------
    st = pd.DataFrame(COV)
    assert len(st) == 268 and (st.status == 'ok').sum() == 225 and (st.status == 'missing').sum() == 43
    miss = st[st.status == 'missing']
    cnt = {'gcs': int(miss.family.str.startswith('gcs').sum()), 'uo': int((miss.family == 'uo').sum()),
           'intake': int((miss.family == 'intake').sum()), 'net': int((miss.family == 'net').sum())}
    assert sum(cnt.values()) == 43
    src = 'jinhu_268_coverage.json'
    add('cov_total', str(len(st)), src, 'len(list)'); add('cov_ok', str((st.status == 'ok').sum()), src, "status == 'ok'")
    add('cov_missing', str(len(miss)), src, "status == 'missing'")
    for k, v in cnt.items():
        add(f'cov_{k}', str(v), src, f"status == 'missing' and family " + ("startswith 'gcs'" if k == 'gcs' else f"== '{k}'"))
    # ---------- additional derived keys used in the text (all computed from the stored values above) ----------
    s12_ = 'f12_full_suite.json'
    for pf, c in (('M4', 'm4'), ('eICU', 'ei')):   # internal minus external Tier-1 point estimates (stored 4-decimal values)
        add(f'drop1_{c}', r(DSC['int_ge1']['auroc'] - DSC[f'{pf}_ge1']['auroc']), s12_, f'discrimination.int_ge1.auroc - discrimination.{pf}_ge1.auroc (4-dp stored values)')
    for c, cc, src in (('int', F11['internal_cal_curve'], 'internal_cal_curve'), ('m4', F11['M4_ext']['cal_curve_ge1'], 'M4_ext.cal_curve_ge1'),
                       ('ei', F11['eICU_ext']['cal_curve_ge1'], 'eICU_ext.cal_curve_ge1')):
        add(f'cal_top_{c}_pred', f'{cc[0][-1]:.4f}', 'f11_full_eval.json', f'{src}[0][-1] (top decile, mean predicted, recalibrated)')
        add(f'cal_top_{c}_obs', f'{cc[1][-1]:.4f}', 'f11_full_eval.json', f'{src}[1][-1] (top decile, observed)')
        assert cc[0][-1] > cc[1][-1]                                       # over-prediction in the top decile in every cohort
    ckn = {'int': T1['Jinhua_test']['ckpt'], 'm4': T1['M4_ext']['ckpt'], 'ei': T1['eICU_ext']['ckpt']}
    for g in (1, 2, 3):
        for pf, c in CK.items():
            add(f'prev{g}_{c}', f"{100 * DSC[f'{pf}_ge{g}']['pos_ckpt'] / ckn[c]:.2f}", s12_, f'derived: discrimination.{pf}_ge{g}.pos_ckpt / table1 ckpt x 100')
    add('eppp_ckpt_cand', f"{ev_ck / SEL['pool']:.1f}", s12_, 'derived: train positive checkpoints (819) / candidate pool (280)')
    add('eppp_pat_cand', f"{ev_pt / SEL['pool']:.2f}", s12_, 'derived: train event patients (155) / candidate pool (280)')
    sh15 = F12['shap_top15']; domsum = {}
    for d in sh15:
        domsum[feature_domain(d['feature'])] = domsum.get(feature_domain(d['feature']), 0) + d['mean_abs']
    cntdom = {}
    for d in sh15:
        cntdom[feature_domain(d['feature'])] = cntdom.get(feature_domain(d['feature']), 0) + 1
    assert sum(cntdom.values()) == 15
    for i, dm in enumerate(FEATURE_DOMAIN_ORDER):
        add(f'shapdom{i}_n', str(cntdom.get(dm, 0)), s12_, f'derived: count of shap_top15 features in domain "{dm}"')
        add(f'shapdom{i}_share', f"{100 * domsum.get(dm, 0) / sum(domsum.values()):.1f}", s12_, f'derived: share of summed mean|SHAP| (top 15) in domain "{dm}"')
    add('shap_sum15', f"{sum(domsum.values()):.3f}", s12_, 'derived: sum of mean|SHAP| over the top 15')
    # ---------- SM3 carry-over (companion package): crosswalk cohort descriptors and vocabulary hits ----------
    cw = cs + 'n8_1_eicu_abi_crosswalk.json (same cohort, companion package)'
    add('cw_visits', n(n81['n_hosp_visits']), cw, 'n_hosp_visits')
    add('cw_age', f"{n81['age_median']:.0f}", cw, 'age_median (13,344-stay crosswalk cohort)')
    add('cw_male', f"{n81['male_pct']:.1f}", cw, 'male_pct (13,344-stay crosswalk cohort)')
    add('cw_charlson', f"{n81['charlson_median']:.0f}", cw, 'charlson_median (13,344-stay crosswalk cohort)')
    for k, v in n81['unittype_top'].items():
        add('ut_' + k.replace(' ', '_').replace('-', '_'), n(v), cw, f'unittype_top.{k}')
    import re as _re
    anc = open(os.path.join(DC, 'p1_SM3_anchor_table.md'), encoding='utf-8').read()
    asrc = cs + 'p1_SM3_anchor_table.md (vocabulary hits as recorded in the companion package SM3; not recomputed here)'
    hit_rows = {}
    for line in anc.splitlines():
        m = _re.match(r'\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([\d,]+(?: / [\d,]+)?)\s*\|$', line)
        if m and not line.startswith('|---') and m.group(1) not in ('ABI category (ICD-10 anchor)', 'Subtype'):
            hit_rows[m.group(1)] = m.group(3)
    hk = {'Traumatic brain injury (S06)': 'hit_tbi', 'Skull fracture (S02)': 'hit_skull', 'Subarachnoid haemorrhage (I60)': 'hit_sah',
          'Intracerebral haemorrhage (I61/I62)': 'hit_ich', 'Ischaemic stroke (I63)': 'hit_is', 'Unspecified stroke (I64)': 'hit_unspec',
          'Anoxic encephalopathy (G93.1)': 'hit_anoxic', 'Encephalitis (G04)': 'hit_enc'}
    assert set(hk) <= set(hit_rows), (set(hk) - set(hit_rows))
    for lab, key in hk.items():
        add(key, hit_rows[lab], asrc, f'anchor table row "{lab}"')
    # ======================= post hoc supplement (f13): frozen f12 prediction parquets only =======================
    import numpy as _np
    F13 = j('f13_posthoc_results.json'); s13 = 'f13_posthoc_results.json'
    M13 = F13['_meta']
    assert F13['repro']['all_ok'] and F13['repro']['n_checks'] == 27
    assert M13['seed'] == 42 and M13['equivalence_margin'] == 0.05
    assert (M13['tier1_layer']['a'], M13['tier1_layer']['b']) == (CAL['a'], CAL['b'])
    add('f13_Bcal', n(M13['B_calibration']), s13, '_meta.B_calibration')
    add('f13_Bprox', n(M13['B_proximity']), s13, '_meta.B_proximity')
    add('f13_Beq', n(M13['B_equivalence']), s13, '_meta.B_equivalence')
    add('f13_checks', str(F13['repro']['n_checks']), s13, 'repro.n_checks (all reproduce the f12 values from the prediction files)')

    # ---- 1/2. calibration bootstrap CI and Brier (Tier 1) ----
    CB = F13['calibration_bootstrap']
    for pf, c in CK.items():
        v = CB[pf]; pt, ci = v['point'], v['ci']; f = f'calibration_bootstrap.{pf}'
        assert v['B'] == v['n_valid'] == M13['B_calibration']
        assert abs(pt['raw_slope'] - CAL[pf]['raw']['slope']) < 6e-4 and abs(pt['recal_slope'] - CAL[pf]['recal']['slope']) < 6e-4
        assert abs(pt['raw_ece'] - CAL[pf]['raw']['ece']) < 6e-5 and abs(pt['recal_ece'] - CAL[pf]['recal']['ece']) < 6e-5
        for k in ('raw_slope', 'recal_slope', 'brier_raw', 'brier_recal', 'bss_raw', 'bss_recal'):
            assert ci[k][0] <= pt[k] <= ci[k][1], (pf, k)
        assert v['n_stays'] == STAYS[pf] and v['n_pos_ckpt'] == DSC[f'{pf}_ge1']['pos_ckpt'] and v['n_event_stays'] == DSC[f'{pf}_ge1']['event_stays']
        assert f"{100 * pt['prevalence']:.2f}" == N[f'prev1_{c}']
        for k in ('raw_slope', 'recal_slope'):
            add(f'cb_{c}_{k}_ci', f"{ci[k][0]:.3f}{EN}{ci[k][1]:.3f}", s13, f + f'.ci.{k}')
        for k in ('raw_ece', 'recal_ece'):
            add(f'cb_{c}_{k}_ci', f"{ci[k][0]:.4f}{EN}{ci[k][1]:.4f}", s13, f + f'.ci.{k}')
        for k in ('raw', 'recal'):
            add(f'br_{c}_{k}', f"{pt['brier_' + k]:.4f}", s13, f + f'.point.brier_{k}')
            add(f'br_{c}_{k}_ci', f"{ci['brier_' + k][0]:.4f}{EN}{ci['brier_' + k][1]:.4f}", s13, f + f'.ci.brier_{k}')
            add(f'bss_{c}_{k}', r(pt['bss_' + k]), s13, f + f'.point.bss_{k}')
            add(f'bss_{c}_{k}_ci', f"{r(ci['bss_' + k][0])} to {r(ci['bss_' + k][1])}", s13, f + f'.ci.bss_{k}')
        add(f'br_{c}_null', f"{pt['brier_null']:.4f}", s13, f + '.point.brier_null (prevalence x (1 - prevalence), cohort prevalence)')
        assert abs(pt['brier_null'] - pt['prevalence'] * (1 - pt['prevalence'])) < 1e-12
        assert abs(pt['bss_recal'] - (1 - pt['brier_recal'] / pt['brier_null'])) < 1e-12
    assert CB['M4']['point']['bss_raw'] < 0 and CB['eICU']['point']['bss_raw'] < 0 < CB['int']['point']['bss_raw']
    assert CB['M4']['ci']['bss_recal'][0] < 0 < CB['M4']['ci']['bss_recal'][1] and CB['eICU']['ci']['bss_recal'][0] > 0 and CB['int']['ci']['bss_recal'][0] < 0

    # ---- 3. proximity standardisation ----
    PX = F13['proximity']; BK = ['0', '6', '12-18', '>=24']; BKN = ['b0', 'b6', 'b12', 'b24']
    for pf, c in (('M4', 'm4'), ('eICU', 'ei')):
        v = PX[pf]; f = f'proximity.{pf}'; pt = v['point']
        assert v['bins_h'] == BK and v['B'] == M13['B_proximity']
        assert round(pt['ge3'], 4) == DSC[f'{pf}_ge3']['auroc'] and round(pt['ge1_raw'], 4) == DSC[f'{pf}_ge1']['auroc'] and round(pt['ge2_raw'], 4) == DSC[f'{pf}_ge2']['auroc']
        for g in (1, 2, 3):
            assert sum(v[f'ge{g}_auc_by_bin'][b]['n_pos'] for b in BK) == DSC[f'{pf}_ge{g}']['pos_ckpt']
            for b, bn in zip(BK, BKN):
                e = v[f'ge{g}_auc_by_bin'][b]; assert e['auroc'] is not None and e['n_pos'] >= 10
                add(f'px_{c}_n{g}_{bn}', n(e['n_pos']), s13, f + f'.ge{g}_auc_by_bin.{b}.n_pos')
                add(f'px_{c}_a{g}_{bn}', f"{e['auroc']:.3f}", s13, f + f'.ge{g}_auc_by_bin.{b}.auroc')
        sh = v['ge3_pos_share_by_bin']; assert abs(sum(sh.values()) - 1) < 1e-9
        for b, bn in zip(BK, BKN):
            add(f'px_{c}_sh_{bn}', f"{100 * sh[b]:.1f}", s13, f + f'.ge3_pos_share_by_bin.{b} x 100')
        assert r(pt['ge3']) == N[f'auc3_{c}'] and r(pt['ge1_raw']) == N[f'auc1_{c}'] and r(pt['ge2_raw']) == N[f'auc2_{c}']
        for g in (1, 2):
            add(f'px_{c}_std{g}', r(pt[f'ge{g}_std']), s13, f + f'.point.ge{g}_std')
            cc = v[f'ge{g}std_ci']; assert cc[0] <= pt[f'ge{g}_std'] <= cc[1]
            add(f'px_{c}_std{g}_ci', f"{r(cc[0])}{EN}{r(cc[1])}", s13, f + f'.ge{g}std_ci')
            dd = v[f'ge3_minus_ge{g}std']; assert dd['n_valid'] == M13['B_proximity'] and abs(dd['diff'] - (pt['ge3'] - pt[f'ge{g}_std'])) < 1e-12
            add(f'px_{c}_d{g}', sg(dd['diff']), s13, f + f'.ge3_minus_ge{g}std.diff')
            add(f'px_{c}_d{g}_ci', f"{r(dd['ci'][0])} to {r(dd['ci'][1])}", s13, f + f'.ge3_minus_ge{g}std.ci')
            add(f'px_{c}_d{g}_p', pfmt(dd['p'], dd['n_valid']), s13, f + f'.ge3_minus_ge{g}std.p' + (' (0.0 -> P < 2/n_valid)' if dd['p'] == 0.0 else ''))
            add(f'px_{c}_d{g}_nv', n(dd['n_valid']), s13, f + f'.ge3_minus_ge{g}std.n_valid')
            assert dd['ci'][0] > 0                                     # the severe-tier advantage keeps an interval above zero after standardisation
        raw31 = pt['ge3'] - pt['ge1_raw']
        assert abs(raw31 - float(N[f'pr31_{c}'].replace(MINUS, '-'))) < 0.005          # point difference vs bootstrap-mean difference of Table 2b
        add(f'px_{c}_raw31', sg(raw31), s13, f + '.point.ge3 - point.ge1_raw (point difference; Table 2b value is the mean of replicate differences)')
        add(f'px_{c}_att', f"{100 * (1 - (pt['ge3'] - pt['ge1_std']) / raw31):.0f}", s13, f + '.derived: 100 x (1 - standardised difference / raw point difference), Tier 2 vs Tier 1')
    assert PX['eICU']['ge3_minus_ge1std']['ci'][0] < 0.01        # lower bound close to zero (text says so)

    # ---- 5. equivalence ----
    EQ = F13['equivalence']; MG = EQ['margin']
    assert MG == 0.05 and EQ['B'] == M13['B_equivalence']
    add('eq_margin', f'{MG:.2f}', s13, 'equivalence.margin (user-specified, post hoc)')
    add('eq_B', n(EQ['B']), s13, 'equivalence.B')

    def eqkeys(tag, v, f):
        assert v['n_valid'] == EQ['B'] and v['p_tost'] > 0
        add(f'eq_{tag}_d', sg(v['delta_obs']), s13, f + '.delta_obs (observed point difference)')
        add(f'eq_{tag}_ci90', f"{r(v['ci90'][0])} to {r(v['ci90'][1])}", s13, f + '.ci90')
        add(f'eq_{tag}_ci95', f"{r(v['ci95'][0])} to {r(v['ci95'][1])}", s13, f + '.ci95')
        add(f'eq_{tag}_pt', f"{v['p_tost']:.3f}", s13, f + '.p_tost (larger one-sided tail proportion beyond the margin)')
        add(f'eq_{tag}_ok', 'yes' if v['equivalent_ci90_within_margin'] else 'no', s13, f + '.equivalent_ci90_within_margin')
        add(f'eq_{tag}_se', f"{v['se']:.3f}", s13, f + '.se (bootstrap)')
        add(f'eq_{tag}_minm', f"{v['min_symmetric_margin_ci90']:.3f}", s13, f + '.min_symmetric_margin_ci90')
        assert v['equivalent_ci90_within_margin'] == (v['ci90'][0] > -MG and v['ci90'][1] < MG)
    for pf, c in CK.items():
        v = EQ['per_cohort'][pf]
        assert abs(v['delta_obs'] - PAR[f'{pf}_ge2_minus_ge1']['delta']) < 0.006, pf
        eqkeys(c, v, f'equivalence.per_cohort.{pf}')
    for key, tag, mem in (('pooled_external', 'pool', ('M4', 'eICU')), ('pooled_all_three', 'all', ('int', 'M4', 'eICU'))):
        v = EQ[key]; f = f'equivalence.{key}'
        eqkeys(tag, v, f)
        assert set(v['weights']) == set(mem) and abs(sum(v['weights'].values()) - 1) < 1e-9 and v['Q_df'] == len(mem) - 1
        add(f'eq_{tag}_Q', f"{v['Q']:.2f}", s13, f + '.Q'); add(f'eq_{tag}_Qp', f"{v['Q_p']:.3f}", s13, f + '.Q_p')
        add(f'eq_{tag}_I2', f"{100 * v['I2']:.1f}", s13, f + '.I2 x 100')
        for m_ in mem:
            add(f'eq_{tag}_w_{CK[m_]}', f"{100 * v['weights'][m_]:.1f}", s13, f + f'.weights.{m_} x 100')
    assert EQ['pooled_external']['equivalent_ci90_within_margin'] is True and EQ['pooled_all_three']['equivalent_ci90_within_margin'] is False
    assert not any(EQ['per_cohort'][pf]['equivalent_ci90_within_margin'] for pf in CK)

    # ---- 4. Tier-2 recalibration views (frozen scores) ----
    T2 = F13['tier2_recal']; LAY = T2['layers']
    for pf, c in (('M4', 'm4'), ('eICU', 'ei')):
        l_ = LAY[pf]; assert l_['n_pos_ckpt'] == DSC[f'{pf}_ge3']['pos_ckpt'] and l_['n_event_stays'] == DSC[f'{pf}_ge3']['event_stays']
        add(f't2l_{c}_a', r(l_['a']), s13, f'tier2_recal.layers.{pf}.a'); add(f't2l_{c}_b', r(l_['b']), s13, f'tier2_recal.layers.{pf}.b')
    t2raw = {}
    for e_ in T2['eval']:
        tc = CK[e_['target']]; sc = CK[e_['layer_from']]; f = f"tier2_recal.eval[{e_['target']}<-{e_['layer_from']}]"
        assert e_['n_pos_ckpt'] == DSC[f"{e_['target']}_ge3"]['pos_ckpt'] and e_['n_event_stays'] == DSC[f"{e_['target']}_ge3"]['event_stays']
        t2raw.setdefault(tc, (e_['before']['slope'], e_['before']['ece']))
        assert t2raw[tc] == (e_['before']['slope'], e_['before']['ece'])
        add(f't2e_{tc}_{sc}_s1', f"{e_['after']['slope']:.3f}", s13, f + '.after.slope'); add(f't2e_{tc}_{sc}_e1', f"{e_['after']['ece']:.4f}", s13, f + '.after.ece')
    for e_ in T2['local_crossfit']:
        tc = CK[e_['target']]; f = f"tier2_recal.local_crossfit[{e_['target']}]"
        assert e_['folds'] == 5 and e_['grouping'] == 'stay_id' and t2raw[tc] == (e_['before']['slope'], e_['before']['ece'])
        add(f't2x_{tc}_s1', f"{e_['after']['slope']:.3f}", s13, f + '.after.slope'); add(f't2x_{tc}_e1', f"{e_['after']['ece']:.4f}", s13, f + '.after.ece')
    for tc, (sl, ec) in t2raw.items():
        add(f't2raw_{tc}_slope', f'{sl:.3f}', s13, f'tier2_recal.eval[{tc}].before.slope (raw Tier-2 output)')
        add(f't2raw_{tc}_ece', f'{ec:.4f}', s13, f'tier2_recal.eval[{tc}].before.ece (raw Tier-2 output)')
    assert set(t2raw) == {'int', 'm4', 'ei'}
    T2THR = (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20)
    t2seen = set()
    for v in T2['thresholds']:
        tc = CK[v['target']]; tk = '%03d' % round(v['thr'] * 100); p = f't2d_{tc}_{tk}_'; f = f"tier2_recal.thresholds[{v['target']}<-{v['layer_from']}, thr {v['thr']}]"
        assert v['layer_from'] == ('eICU' if v['target'] == 'M4' else 'M4') and v['n_event_stays'] == DSC[f"{v['target']}_ge3"]['event_stays']
        assert abs(v['capture'] - v['n_captured'] / v['n_event_stays']) <= 0.0006 and abs(v['pct_stays_alerted'] - 100 * v['n_alert_stays'] / STAYS[v['target']]) <= 0.05
        t2seen.add((tc, v['thr']))
        add(p + 'cap', r(100 * v['capture'], 1), s13, f + '.capture x 100'); add(p + 'ncap', n(v['n_captured']), s13, f + '.n_captured')
        add(p + 'nev', n(v['n_event_stays']), s13, f + '.n_event_stays')
        add(p + 'lead', gfmt(v['lead_median_h']) if v['lead_median_h'] is not None else DASH, s13, f + '.lead_median_h (dash = no captured stay)')
        add(p + 'iqr', f"{gfmt(v['lead_iqr'][0])}{EN}{gfmt(v['lead_iqr'][1])}" if v['lead_iqr'] else DASH, s13, f + '.lead_iqr (dash = no captured stay)')
        add(p + 'nne', f"{v['NNE']:.2f}" if v['NNE'] is not None else DASH, s13, f + '.NNE (dash = no captured stay)')
        add(p + 'fa', f"{v['FA_per_100ptd']:.2f}", s13, f + '.FA_per_100ptd'); add(p + 'pct', f"{v['pct_stays_alerted']:.1f}", s13, f + '.pct_stays_alerted')
        add(p + 'nal', n(v['n_alert_stays']), s13, f + '.n_alert_stays')
    assert t2seen == {(tc, t) for tc in ('m4', 'ei') for t in T2THR}
    for thr_ in T2THR:
        add('t2thr_%03d' % round(thr_ * 100), f'{thr_:.2f}', s13, 'tier2_recal.thresholds threshold label')
    add('t2thr_lo', f'{T2THR[0]:.2f}', s13, 'Tier-2 threshold view, lowest threshold (exploratory)'); add('t2thr_hi', f'{T2THR[-1]:.2f}', s13, 'Tier-2 threshold view, highest threshold')
    add('t2thr_expl_hi', f'{T2THR[2]:.2f}', s13, 'Tier-2 threshold view, highest exploratory threshold (0.01-0.03 added to the pre-specified 0.05-0.20 grid)')

    # ---- 6. utility curve (eICU requested; MIMIC-IV extra) and FA review ----
    UTHR = [round(0.03 + 0.01 * i, 2) for i in range(28)]
    for nm, tg, pf, c in (('utility_eICU', 'e', 'eICU', 'ei'), ('utility_M4', 'm', 'M4', 'm4')):
        rows_ = F13[nm]; assert [round(x['thr'], 2) for x in rows_] == UTHR
        csv = pd.read_csv(os.path.join(D, f'f13_utility_curve_{pf}.csv')); assert len(csv) == 28
        for k_ in ('thr', 'capture', 'NNE', 'FA_per_100ptd', 'pct_stays_alerted', 'n_captured', 'n_alert_stays', 'lead_gt48h_n', 'n_event_stays'):
            assert _np.allclose(csv[k_].values.astype(float), _np.array([x[k_] for x in rows_], dtype=float)), (pf, k_)
        caps = [x['capture'] for x in rows_]; pcts = [x['pct_stays_alerted'] for x in rows_]
        assert caps == sorted(caps, reverse=True) and pcts == sorted(pcts, reverse=True)
        for x in rows_:
            tt = '%02d' % round(x['thr'] * 100); p = f'u{tg}_{tt}_'; f = f"{nm}[thr {x['thr']}]"
            assert x['n_event_stays'] == DSC[f'{pf}_ge1']['event_stays'] and abs(x['capture'] - x['n_captured'] / x['n_event_stays']) <= 0.0006
            assert abs(x['lead_gt48h_frac'] - x['lead_gt48h_n'] / x['n_captured']) <= 0.0006
            add(p + 'cap', r(100 * x['capture'], 1), s13, f + '.capture x 100'); add(p + 'ncap', n(x['n_captured']), s13, f + '.n_captured')
            add(p + 'nev', n(x['n_event_stays']), s13, f + '.n_event_stays'); add(p + 'lead', gfmt(x['lead_median_h']), s13, f + '.lead_median_h')
            add(p + 'iqr', f"{gfmt(x['lead_iqr'][0])}{EN}{gfmt(x['lead_iqr'][1])}", s13, f + '.lead_iqr'); add(p + 'nne', f"{x['NNE']:.2f}", s13, f + '.NNE')
            add(p + 'fa', f"{x['FA_per_100ptd']:.2f}", s13, f + '.FA_per_100ptd'); add(p + 'pct', f"{x['pct_stays_alerted']:.1f}", s13, f + '.pct_stays_alerted')
            add(p + 'nal', n(x['n_alert_stays']), s13, f + '.n_alert_stays')
            add(p + 'l48n', n(x['lead_gt48h_n']), s13, f + '.lead_gt48h_n')
            add(p + 'l48', r(100 * x['lead_gt48h_n'] / x['n_captured'], 1), s13, f + '.lead_gt48h_n / n_captured x 100')
            for t_, tk_ in TK.items():                      # the utility curve must reproduce the f12 deployment block at its five thresholds
                if abs(x['thr'] - float(t_)) < 1e-9:
                    for k_ in ('cap', 'lead', 'iqr', 'nne', 'fa', 'pct', 'ncap', 'nev', 'nal'):
                        assert N[p + k_] == N[f'dp_{c}_{tk_}_{k_}'], (pf, t_, k_)
        add(f'u{tg}_cap_lo', r(100 * min(caps), 1), s13, f'min over {nm}.capture x 100 (thr 0.30)')
        add(f'u{tg}_cap_hi', r(100 * max(caps), 1), s13, f'max over {nm}.capture x 100 (thr 0.03)')
        add(f'u{tg}_nne_hi', f"{max(x['NNE'] for x in rows_):.2f}", s13, f'max over {nm}.NNE'); add(f'u{tg}_nne_lo', f"{min(x['NNE'] for x in rows_):.2f}", s13, f'min over {nm}.NNE')
        add(f'u{tg}_pct_hi', f"{max(pcts):.1f}", s13, f'max over {nm}.pct_stays_alerted'); add(f'u{tg}_pct_lo', f"{min(pcts):.1f}", s13, f'min over {nm}.pct_stays_alerted')
    for thr_ in UTHR:
        add('uthr_%02d' % round(thr_ * 100), f'{thr_:.2f}', s13, 'utility sweep threshold label')
    add('u_thr_lo', f'{UTHR[0]:.2f}', s13, 'utility sweep lower threshold (as f12 np.arange(0.03, 0.301, 0.01))')
    add('u_thr_hi', f'{UTHR[-1]:.2f}', s13, 'utility sweep upper threshold')
    add('u_nthr', str(len(UTHR)), s13, 'number of thresholds in the sweep')
    add('u_step', f'{UTHR[1] - UTHR[0]:.2f}', s13, 'utility sweep step (np.arange step 0.01, as f12)')
    FA = F13['fa_review']
    for pf, c in (('M4', 'm4'), ('eICU', 'ei')):
        v = FA[pf]; f = f'fa_review.{pf}'; dk = f'dp_{c}_015_'
        assert v['thr'] == 0.15 and n(v['n_captured']) == N[dk + 'ncap'] and n(v['n_alert_stays']) == N[dk + 'nal'] and n(v['n_event_stays']) == N[dk + 'nev']
        assert v['n_event_stays'] + v['n_event_free_stays'] == STAYS[pf] and v['n_alert_stays'] >= v['n_false_alert_stays'] + v['n_captured']
        assert v['near_criteria_creatinine_share'] is None
        add(f'fa_{c}_evfree', n(v['n_event_free_stays']), s13, f + '.n_event_free_stays'); add(f'fa_{c}_false', n(v['n_false_alert_stays']), s13, f + '.n_false_alert_stays (alerted event-free stays)')
        add(f'fa_{c}_false_pct_evfree', r(100 * v['n_false_alert_stays'] / v['n_event_free_stays'], 1), s13, f + '.derived: n_false_alert_stays / n_event_free_stays x 100')
        add(f'fa_{c}_false_pct_alert', r(100 * v['n_false_alert_stays'] / v['n_alert_stays'], 1), s13, f + '.derived: n_false_alert_stays / n_alert_stays x 100')
        add(f'fa_{c}_evalert', n(v['n_alert_stays'] - v['n_false_alert_stays']), s13, f + '.derived: n_alert_stays - n_false_alert_stays (alerted event stays)')
        add(f'fa_{c}_l48n', n(v['lead_gt48h_n']), s13, f + '.lead_gt48h_n'); add(f'fa_{c}_l48', r(100 * v['lead_gt48h_n'] / v['n_captured'], 1), s13, f + '.derived: lead_gt48h_n / n_captured x 100')
        assert abs(v['lead_gt48h_frac'] - v['lead_gt48h_n'] / v['n_captured']) <= 0.0006
        assert N[f'u{"m" if pf == "M4" else "e"}_15_l48n'] == N[f'fa_{c}_l48n']

    # ---- 7/8. grid descriptors from the prediction files; MIMIC-IV stay-subset descriptors (companion package) ----
    DS = F13['descriptors']
    for pf, c in CK.items():
        v = DS[pf]; f = f'descriptors.{pf}'
        assert v['stays'] == STAYS[pf] and n(v['ckpt']) == N[f"t1_{'test' if c == 'int' else c}_ckpt"] and v['first_ckpt_hr_min'] == v['first_ckpt_hr_max'] == 24.0
        add(f'ds_{c}_cps', f"{gfmt(v['ckpt_per_stay_median'])} ({gfmt(v['ckpt_per_stay_q1'])}{EN}{gfmt(v['ckpt_per_stay_q3'])})", s13, f + '.ckpt_per_stay median (q1-q3)')
        add(f'ds_{c}_last', f"{gfmt(v['last_ckpt_hr_median'])} ({gfmt(v['last_ckpt_hr_q1'])}{EN}{gfmt(v['last_ckpt_hr_q3'])})", s13, f + '.last_ckpt_hr median (q1-q3)')
        add(f'ds_{c}_evpct', r(v['event_stay_pct_ge1'], 1), s13, f + '.event_stay_pct_ge1')
        assert r(v['event_stay_pct_ge1'], 1) == N[f"t1_{'test' if c == 'int' else c}_stayrate1"] or abs(v['event_stay_pct_ge1'] - float(N[f"t1_{'test' if c == 'int' else c}_stayrate1"])) < 0.06
    MS = F13['m4_membership_check']; SP = MS['strata_from_predictions']
    tr, te = f3['rows'][0], f3['rows'][1]
    assert tr['cohort'].startswith('M4 training') and te['cohort'].startswith('M4 internal test')
    assert MS['all_contained'] and MS['M4_stays'] == T1['M4_ext']['stays'] == tr['stays'] + te['stays'] and MS['companion_test_stays'] == te['stays'] and MS['complement_stays'] == tr['stays']
    cm_ = cs + 'f3_table1.json M4 rows (same cohort, companion package)'
    for tag, row, sp in (('a', tr, SP['complement_subset']), ('b', te, SP['test_subset'])):
        assert sp['stays'] == row['stays'] and sp['ckpt'] == row['checkpoints']                           # membership verified through stay_id against the prediction file
        for g in (1, 2, 3):
            assert sp[f'event_stays_ge{g}'] == row[f'ge{g}_event_stays'] and sp[f'pos_ckpt_ge{g}'] == row[f'ge{g}_pos_ckpt'], (tag, g)
        a_ = row['age_median_iqr']; c_ = row['charlson_median_iqr']; b_ = row['cr_base_median_iqr']
        add(f'm4{tag}_stays', n(row['stays']), cm_, 'stays (verified against f12_preds_M4_ge1.parquet by stay_id)')
        add(f'm4{tag}_subj', n(row['subjects']), cm_, 'subjects'); add(f'm4{tag}_ckpt', n(row['checkpoints']), cm_, 'checkpoints (verified against the prediction file)')
        add(f'm4{tag}_age', f'{a_[0]:.0f} ({a_[1]:.0f}{EN}{a_[2]:.0f})', cm_, 'age_median_iqr'); add(f'm4{tag}_male', f"{row['male_pct']:.1f}", cm_, 'male_pct')
        add(f'm4{tag}_charlson', f'{c_[0]:.0f} ({c_[1]:.0f}{EN}{c_[2]:.0f})', cm_, 'charlson_median_iqr')
        add(f'm4{tag}_crbase', f'{b_[0]:.1f} ({b_[1]:.1f}{EN}{b_[2]:.1f})', cm_, 'cr_base_median_iqr')
        for g in (1, 2, 3):
            add(f'm4{tag}_evs{g}', n(row[f'ge{g}_event_stays']), cm_, f'ge{g}_event_stays (verified against the prediction file)')
            add(f'm4{tag}_pos{g}', n(row[f'ge{g}_pos_ckpt']), cm_, f'ge{g}_pos_ckpt (verified against the prediction file)')
    for g in (1, 2, 3):
        assert tr[f'ge{g}_event_stays'] + te[f'ge{g}_event_stays'] == DSC[f'M4_ge{g}']['event_stays'] and tr[f'ge{g}_pos_ckpt'] + te[f'ge{g}_pos_ckpt'] == DSC[f'M4_ge{g}']['pos_ckpt']
    assert tr['stays'] + te['stays'] == int(N['t1_m4_stays'].replace(',', '')) and tr['checkpoints'] + te['checkpoints'] == int(N['t1_m4_ckpt'].replace(',', ''))
    add('t1_m4_subj', n(tr['subjects'] + te['subjects']), cm_, 'derived: subset A subjects + subset B subjects (patient-level strata, disjoint by construction; stays and checkpoints of the two strata sum to the pooled grid, asserted)')
    add('m4ab_checks', '14', s13, 'm4_membership_check.strata_from_predictions: stays, checkpoints, 3 x event stays and 3 x positive checkpoints per subset, all equal to the companion rows')

    # ---- hospital / SHAP file QC (supplied files) ----
    HQ = F13['hospital_file']
    assert HQ['matches_f12_json'] and str(HQ['n']) == N['hosp_n'] and f"{HQ['median']:.3f}" == N['hosp_med'] and str(HQ['n_below_060']) == N['hosp_lt60']
    add('hq_ge70', n(HQ['n_ge_070']), s13, 'hospital_file.n_ge_070'); add('hq_ge70_pct', r(100 * HQ['n_ge_070'] / HQ['n'], 1), s13, 'derived: n_ge_070 / n x 100')
    add('hq_ge80', n(HQ['n_ge_080']), s13, 'hospital_file.n_ge_080'); add('hq_ge80_pct', r(100 * HQ['n_ge_080'] / HQ['n'], 1), s13, 'derived: n_ge_080 / n x 100')
    SQ = F13['shap_file']
    assert SQ['matches_f12_json_top15'] and SQ['same_stay_order'] and n(SQ['rows']) == N['shap_n'] and str(SQ['n_features']) == N['n_feat'] and SQ['rows'] == SQ['X_rows']
    add('sq_rows', n(SQ['rows']), s13, 'shap_file.rows'); add('sq_feats', str(SQ['n_features']), s13, 'shap_file.n_features')
    EXF = F13['exemplar_file']
    add('ex_rows', n(EXF['rows']), s13, 'exemplar_file.rows'); add('ex_stays', str(EXF['stays']), s13, 'exemplar_file.stays')
    add('f13_inputs', str(len(F13['input_sha256'])), s13, 'len(input_sha256): supplied input files hashed')

    # constants documented in scripts (verified by audit_r11.py against the shipped script text)
    return N, L, dict(F10=F10, F11=F11, F12=F12, COV=COV, T1S=T1S, n3=n3, n81=n81, n83=n83, f3=f3, cohort=cohort, STAYS=STAYS, DSC=DSC, F13=F13)


FEATURE_DOMAIN_ORDER = ['Creatinine kinetics', 'Haemodynamics and vital signs', 'Treatment exposure', 'Demographics, comorbidity and time',
                        'Other laboratory and acid\u2013base']


def feature_domain(f):
    """Clinical domain of a feature name (rule inherited from the companion package; GCS / fluid-balance families do not occur in the 166 set)."""
    if f.startswith('cr_'): return 'Creatinine kinetics'
    if f.startswith(('sbp', 'hr_rate', 'rr_', 'spo2')): return 'Haemodynamics and vital signs'
    if f in ('age', 'charlson', 'male', 't_hr'): return 'Demographics, comorbidity and time'
    if f.startswith(('norepi', 'propofol', 'dex', 'mido', 'hts', 'vent')): return 'Treatment exposure'
    return 'Other laboratory and acid\u2013base'


def readme_asserts(N):
    """README section 2 / 3 numbers, typed verbatim from README_BIOMNI_ROUND11.md, compared with the ledger-formatted values.
    Key -> (README text, README location)."""
    exp = {
        # main discrimination table (README 2)
        'auc1_int': '0.815', 'ci1_int': '0.754' + EN + '0.872', 'auc1_m4': '0.769', 'ci1_m4': '0.748' + EN + '0.791',
        'auc1_ei': '0.777', 'ci1_ei': '0.759' + EN + '0.795', 'auc3_int': '0.995', 'auc3_m4': '0.922', 'ci3_m4': '0.882' + EN + '0.954',
        'auc3_ei': '0.878', 'ci3_ei': '0.791' + EN + '0.953',
        # paired
        'pr21_m4': '+0.022', 'pr21_m4_pnum': '0.484', 'pr21_ei': MINUS + '0.012', 'pr21_ei_pnum': '0.638',
        'pr31_m4': '+0.153', 'pr31_m4_p': 'P < 0.002', 'pr31_ei': '+0.101', 'pr31_ei_pnum': '0.012',
        # gradient
        'sev_s12_m4': '0.763', 'sev_s3_m4': '0.862', 'sev_s12_ei': '0.768', 'sev_s3_ei': '0.891',
        # deployment eICU thr 0.15
        'dp_ei_015_cap': '34.8', 'dp_ei_015_lead': '18', 'dp_ei_015_nne': '6.41', 'dp_ei_015_pct': '18.4', 'dp_ei_015_fa': '4.82',
        # hospitals / MK / SHAP
        'hosp_n': '50', 'hosp_med': '0.754', 'hosp_lt60': '2', 'mk_tau': '1.000', 'mk_p_asym': 'P = 0.003',
        'shap1': 'cr_ratio_base', 'shap2': 'age', 'shap3': 'cr_last48', 'shap4': 'charlson', 'shap5': 'sbp_locf',
        # EPPP / selection
        'eppp_ckpt': '4.9', 'eppp_pat': '0.93', 'eppp_params': '166', 'n_cand': '280', 'n_feat': '166',
        # cohorts
        't1_m4_stays': '7,442', 't1_m4_ckpt': '75,638', 't1_ei_stays': '9,071', 't1_ei_ckpt': '84,866', 'ei_hosp': '171',
        'jh_stays': '1,033', 'jh_ckpt': '12,483',
        # calibration layer (Tier 1) and internal Tier 2 events
        'recal_a': '0.005067', 'recal_b': '0.268467', 'evs3_int': '3',
    }
    bad = {k: (N[k], v) for k, v in exp.items() if N[k] != v}
    assert not bad, bad
    return exp


if __name__ == '__main__':
    N, L, _ = load(sys.argv[1], sys.argv[2])
    exp = readme_asserts(N)
    df = pd.DataFrame(L); df['in_README'] = df.key.isin(exp)
    df.to_csv(sys.argv[3], index=False)
    print(len(N), 'ledger entries;', len(exp), 'README values asserted: ALL MATCH')
