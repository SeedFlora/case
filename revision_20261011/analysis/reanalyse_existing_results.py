"""Camera-ready descriptive reanalysis of frozen participant metrics.

No fitting or predictions are performed. A participant is the resampling unit:
all three repeated held-out seeds and every draw in a family are averaged first.
The intervals condition on the fitted cross-validation models, chosen protocol,
and finite fault catalogue; they do not capture refitting/tuning uncertainty.
Cross-validation predictions share training sets, so 30 observations must not
be called 30 independent model fits. All additional tests are post hoc.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FAMILIES = ['clean', 'missing_random_1', 'missing_random_2',
            'missing_random_4', 'wearable', 'corrupt_all']
METRICS = ['mean_ccc', 'ccc_valence', 'ccc_arousal']
REPORT_METRICS = METRICS + ['mean_mae', 'mae_valence', 'mae_arousal']
B = 10000
SEED = 13


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    result = np.empty(len(values))
    running = 0.
    for i, j in enumerate(order):
        running = max(running, (len(values) - i) * values[j])
        result[j] = min(1., running)
    return result


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(SEED)
    boot = values[rng.randint(0, len(values), (B, len(values)))].mean(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def describe(values):
    lo, hi = interval(values)
    return dict(n_participants=len(values), mean=float(np.mean(values)),
                sd_between_participants=float(np.std(values, ddof=1)),
                ci95_low=float(lo), ci95_high=float(hi))


def paired(scores, baseline, proposed, **labels):
    a, b = scores[baseline], scores[proposed]
    if not a.index.equals(b.index) or len(a) < 5:
        raise ValueError('Expected identical participant pairs, at least five')
    diff = b.to_numpy() - a.to_numpy()
    if np.all(diff == 0):
        w,p,r=0.,1.,0.
    else:
        test = stats.wilcoxon(diff, zero_method='wilcox', alternative='two-sided')
        w,p=float(test.statistic),float(test.pvalue)
        ranks = stats.rankdata(np.abs(diff[diff != 0]))
        r = (ranks[diff[diff != 0] > 0].sum() -
             ranks[diff[diff != 0] < 0].sum()) / ranks.sum()
    desc = describe(diff)
    desc['mean_difference'] = desc.pop('mean')
    return dict(labels, baseline_model=baseline, proposed_model=proposed,
                baseline_mean=float(a.mean()), proposed_mean=float(b.mean()),
                **desc, median_difference=float(np.median(diff)),
                p_value=p, wilcoxon_W=w,
                effect_size_rank_biserial=float(r),
                n_participants_improved=int((diff > 0).sum()))


def format_ci(row, difference=False):
    key = 'mean_difference' if difference else 'mean'
    pattern = '+.3f' if difference else '.3f'
    return (f"{row[key]:{pattern}} [{row['ci95_low']:{pattern}}, "
            f"{row['ci95_high']:{pattern}}]")


def p_text(p):
    return f'{p:.4f}' if p >= .001 else f'{p:.2g}'


def write_tex_tables(out, family_summaries, clean_summaries, gates, original):
    names = dict(naive='Training mean', ridge='Ridge', lightgbm='LightGBM',
                 early_fusion='Early fusion, no MD', late_fusion='Equal weights, no MD',
                 reliability_gated='Quality gate, MD', abl_no_quality='Constant-input gate, MD',
                 abl_no_moddropout='Quality gate, no MD', abl_equal_weights='Equal weights, MD')
    model_order = list(names)
    fam_names = dict(clean='Clean', missing_random_1='Missing 1', missing_random_2='Missing 2',
                     missing_random_4='Missing 4', wearable='Wearable', corrupt_all='Corrupted')
    clean_lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
                   r'Model & CCC valence & CCC arousal & mCCC & MAE \\', r'\midrule']
    for model in model_order:
        rows = clean_summaries[clean_summaries.model.eq(model)].set_index('metric')
        cells = [format_ci(rows.loc[m]) for m in
                 ['ccc_valence','ccc_arousal','mean_ccc','mean_mae']]
        clean_lines.append(names[model] + ' & ' + ' & '.join(cells) + r' \\')
    clean_lines.extend([r'\bottomrule', r'\end{tabular}'])
    (out / 'table_clean_pointwise_ci.tex').write_text('\n'.join(clean_lines)+'\n',encoding='utf-8')

    # Split six families into two four-column tables to preserve legibility.
    for filename, fams in [('table_missing_pointwise_ci.tex',FAMILIES[:4]),
                           ('table_wearable_corrupt_pointwise_ci.tex',FAMILIES[4:])]:
        lines = [r'\begin{tabular}{l' + 'r'*len(fams) + '}', r'\toprule',
                 'Model & ' + ' & '.join(fam_names[f] for f in fams) + r' \\', r'\midrule']
        for model in model_order:
            rows = family_summaries[(family_summaries.model.eq(model)) &
                                    family_summaries.metric.eq('mean_ccc')].set_index('family')
            lines.append(names[model]+' & '+' & '.join(format_ci(rows.loc[f]) for f in fams)+r' \\')
        lines.extend([r'\bottomrule',r'\end{tabular}'])
        (out/filename).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for filename, df, pcol in [
        ('table_gate_fixed_pointwise_ci.tex',gates[gates.metric.eq('mean_ccc')],
         'p_value_holm_within_metric_six'),
        ('table_historical_early_pointwise_ci.tex',original,'p_value_holm_six')]:
        lines=[r'\begin{tabular}{lrrr}',r'\toprule',
               r'Family & $\Delta$mCCC [95\% CI] & $p$ raw & $p$ Holm \\',r'\midrule']
        for _, row in df.iterrows():
            lines.append(fam_names[row.family]+' & '+format_ci(row,True)+' & '+
                         p_text(row.p_value)+' & '+p_text(row[pcol])+r' \\')
        lines.extend([r'\bottomrule',r'\end{tabular}'])
        (out/filename).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject-metrics',type=Path,default=ROOT/'results/subject_metrics.csv')
    ap.add_argument('--out-dir',type=Path,default=OUT)
    ap.add_argument('--analysis-origin',choices=['historical','retrospective'],default='historical')
    ap.add_argument('--allow-partial-qa',action='store_true',
                    help='Allow incomplete nested results solely in a separate layout QA directory.')
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if args.allow_partial_qa and out == OUT.resolve():
        raise ValueError('Partial QA requires a separate --out-dir')
    out.mkdir(parents=True,exist_ok=True)
    data = pd.read_csv(args.subject_metrics)
    keys = ['seed', 'k', 'model', 'condition', 'subject_id']
    assert not data.duplicated(keys).any()
    n_subjects = data.subject_id.nunique()
    n_seeds = data.seed.nunique()
    n_folds = data.fold.nunique()
    assert data.condition.nunique() == 55
    assert data.model.nunique() == 9
    if not args.allow_partial_qa:
        assert n_subjects == 30 and n_seeds == 3 and n_folds == 15
        assert len(data) == 30 * 3 * 55 * 9
    else:
        assert n_subjects >= 5
        assert data.groupby(['model','condition']).size().nunique() == 1
    assert np.isfinite(data[REPORT_METRICS].to_numpy(dtype=float)).all()
    data['raw_family'] = data['family'].fillna('clean')
    data['family'] = data.raw_family
    missing = data.raw_family.eq('missing_random')
    data.loc[missing, 'family'] = ('missing_random_' +
        data.loc[missing, 'level'].astype(int).astype(str))
    data.loc[data.raw_family.str.startswith('corrupt_'), 'family'] = 'corrupt_all'
    # Exactly one held-out fold per participant per seed and condition/model.
    assert data.groupby(['subject_id','seed','condition','model']).size().eq(1).all()

    gate_rows, historical, factorial, summaries = [], [], [], []
    for fam in FAMILIES:
        family = data[data.family.eq(fam)]
        for metric in REPORT_METRICS:
            scores = family.groupby(['subject_id','model'])[metric].mean().unstack()
            for model in scores.columns:
                summaries.append(dict(family=fam, model=model, metric=metric,
                                      **describe(scores[model].to_numpy())))
            if metric not in METRICS:
                continue
            gate_rows.append(paired(scores, 'abl_equal_weights', 'reliability_gated',
                                    family=fam, metric=metric, analysis='post_hoc_gate'))
            if metric == 'mean_ccc':
                historical.append(paired(scores, 'early_fusion', 'reliability_gated',
                    family=fam, metric=metric, analysis=args.analysis_origin+'_six_family_gate_vs_early'))
                for base, prop, contrast in [
                    ('late_fusion','abl_no_moddropout','gate_at_dropout_0'),
                    ('abl_equal_weights','reliability_gated','gate_at_dropout_0.2'),
                    ('late_fusion','abl_equal_weights','dropout_without_gate'),
                    ('abl_no_moddropout','reliability_gated','dropout_with_gate'),
                    ('abl_no_quality','reliability_gated','quality_vector_vs_constant')]:
                    factorial.append(paired(scores, base, prop, family=fam,
                        metric=metric, contrast=contrast, analysis='post_hoc_factorial'))
                interaction = ((scores.reliability_gated - scores.abl_equal_weights) -
                               (scores.abl_no_moddropout - scores.late_fusion))
                desc = describe(interaction.to_numpy())
                desc['mean_difference_in_differences'] = desc.pop('mean')
                factorial.append(dict(family=fam, metric=metric,
                    contrast='gate_by_dropout_interaction', analysis='post_hoc_factorial',
                    **desc, p_value=1. if np.all(interaction.to_numpy()==0) else float(stats.wilcoxon(interaction).pvalue)))

    original = pd.DataFrame(historical)
    original['p_value_holm_six'] = holm(original.p_value.to_numpy())
    original.to_csv(out / 'original_six_hypotheses.csv', index=False)
    gates = pd.DataFrame(gate_rows)
    # Main gate mCCC contrasts are a new post hoc six-family group, separately
    # labelled from the original gate-versus-early comparison family.
    gates['p_value_holm_within_metric_six'] = np.nan
    for metric in METRICS:
        ix = gates.metric.eq(metric)
        gates.loc[ix, 'p_value_holm_within_metric_six'] = holm(gates.loc[ix,'p_value'])
    gates.to_csv(out / 'gate_vs_fixed_weights_families.csv', index=False)
    pd.DataFrame(factorial).to_csv(out / 'factorial_and_quality_contrasts.csv', index=False)
    family_summaries = pd.DataFrame(summaries)
    family_summaries.to_csv(out / 'family_model_target_intervals.csv', index=False)
    clean_summaries = []
    for metric in ['ccc_valence','ccc_arousal','mean_ccc','mae_valence','mae_arousal','mean_mae']:
        scores = data[data.family.eq('clean')].groupby(['subject_id','model'])[metric].mean().unstack()
        for model in scores.columns:
            clean_summaries.append(dict(model=model,metric=metric,
                                       **describe(scores[model].to_numpy())))
    clean_summaries=pd.DataFrame(clean_summaries)
    clean_summaries.to_csv(out/'clean_model_target_metric_intervals.csv',index=False)

    condition_summaries, condition_pairs = [], []
    for condition, chunk in data.groupby('condition', sort=False):
        for metric in REPORT_METRICS:
            scores = chunk.groupby(['subject_id','model'])[metric].mean().unstack()
            for model in scores.columns:
                condition_summaries.append(dict(condition=condition,
                    family=chunk.family.iloc[0], raw_family=chunk.raw_family.iloc[0],
                    level=chunk.level.iloc[0], model=model, metric=metric,
                    **describe(scores[model].to_numpy())))
            if metric not in METRICS:
                continue
            condition_pairs.append(paired(scores, 'abl_equal_weights', 'reliability_gated',
                condition=condition, family=chunk.family.iloc[0],
                raw_family=chunk.raw_family.iloc[0], level=chunk.level.iloc[0],
                metric=metric, analysis='post_hoc_condition'))
    pd.DataFrame(condition_summaries).to_csv(out / 'all55_condition_model_target_intervals.csv', index=False)
    conditions = pd.DataFrame(condition_pairs)
    conditions['p_value_holm_55_within_metric'] = np.nan
    for metric in METRICS:
        ix = conditions.metric.eq(metric)
        conditions.loc[ix,'p_value_holm_55_within_metric'] = holm(conditions.loc[ix,'p_value'])
    conditions.to_csv(out / 'all55_gate_vs_fixed_weights.csv', index=False)

    corrupt_summaries, corrupt_pairs = [], []
    corrupted = data[data.family.eq('corrupt_all')]
    for (kind, severity), chunk in corrupted.groupby(['raw_family','level'], sort=False):
        for metric in REPORT_METRICS:
            scores = chunk.groupby(['subject_id','model'])[metric].mean().unstack()
            for model in scores.columns:
                corrupt_summaries.append(dict(corruption_type=kind.removeprefix('corrupt_'),
                    severity=severity, model=model, metric=metric,
                    **describe(scores[model].to_numpy())))
            if metric not in METRICS:
                continue
            corrupt_pairs.append(paired(scores,'abl_equal_weights','reliability_gated',
                corruption_type=kind.removeprefix('corrupt_'), severity=severity,
                metric=metric, analysis='post_hoc_corruption_type_severity'))
    pd.DataFrame(corrupt_summaries).to_csv(out / 'corruption_type_severity_model_target_intervals.csv', index=False)
    corrupt_df = pd.DataFrame(corrupt_pairs)
    for metric in METRICS:
        ix = corrupt_df.metric.eq(metric)
        corrupt_df.loc[ix,'p_value_holm_15_within_metric'] = holm(corrupt_df.loc[ix,'p_value'])
    corrupt_df.to_csv(out / 'corruption_type_severity_gate_contrasts.csv', index=False)

    main_gate = gates[gates.metric.eq('mean_ccc')]
    main_cond = conditions[conditions.metric.eq('mean_ccc')]
    information = dict(
        input=str(args.subject_metrics.resolve()),analysis_origin=args.analysis_origin,
        input_sha256=hashlib.sha256(args.subject_metrics.read_bytes()).hexdigest(),
        software_versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        rows=len(data), n_subjects=int(n_subjects), n_folds=int(n_folds),
        n_seeds=int(n_seeds), n_models=9, n_conditions=55,partial_layout_qa=args.allow_partial_qa,
        bootstrap=dict(method='participant percentile', replicates=B, seed=SEED,
                       confidence=.95, simultaneous=False),
        repeated_measurement_handling='Mean within participant across 3 seeds and within-family draws before resampling; pair identical subject IDs.',
        limitations='Conditional on frozen trained models/protocol/fault catalogue. Shared cross-validation training sets induce dependence; refitting/tuning and population uncertainty not captured. Additional tests post hoc; no equivalence margin or formal power calculation.',
        gate_mean_positive_conditions=int((main_cond.mean_difference > 0).sum()),
        gate_mean_negative_conditions=int((main_cond.mean_difference < 0).sum()),
        gate_raw_significant_conditions=int((main_cond.p_value < .05).sum()),
        gate_holm55_significant_conditions=int((main_cond.p_value_holm_55_within_metric < .05).sum()),
        gate_family_ci_halfwidths=((main_gate.ci95_high-main_gate.ci95_low)/2).tolist(),
        original_comparison_note='Early fusion was selected from the historical outer clean test scores, so historical comparisons are exploratory; nested rerun on the same known dataset remains retrospective.',
        gate_family_comparisons=main_gate.to_dict(orient='records'),
        historical_early_comparisons=original.to_dict(orient='records'))
    (out / 'reanalysis_metadata.json').write_text(json.dumps(information, indent=2),encoding='utf-8')
    write_tex_tables(out,family_summaries,clean_summaries,gates,original)
    print(original[['family','mean_difference','p_value','p_value_holm_six']].to_string(index=False))
    print(main_gate[['family','baseline_mean','proposed_mean','mean_difference','ci95_low','ci95_high','p_value','p_value_holm_within_metric_six']].to_string(index=False))
    print(json.dumps(information, indent=2))


if __name__ == '__main__':
    main()
