"""Independently recompute participant CCC/MAE from held-out prediction files."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
METRICS = ['ccc_valence', 'ccc_arousal', 'mean_ccc',
           'mae_valence', 'mae_arousal', 'mean_mae']


def concordance(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    denominator = (np.mean(target_centered**2) + np.mean(prediction_centered**2)
                   + (target.mean() - prediction.mean())**2)
    if denominator == 0:
        return float(np.isclose(target.mean(), prediction.mean()))
    return float(2 * np.mean(target_centered * prediction_centered) / denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    files = sorted((REV/'results').glob('predictions_*.parquet'))
    if not args.allow_partial and len(files) != 15:
        raise ValueError('Final validation requires all 15 prediction files')
    reports = []
    for path in files:
        fold = path.stem.removeprefix('predictions_')
        predictions = pd.read_parquet(path)
        stored = pd.read_csv(REV/'results'/f'subject_metrics_{fold}.csv')
        assert predictions['fold'].nunique() == 1
        assert predictions['subject_id'].nunique() == 6
        assert predictions['model'].nunique() == 9
        assert predictions['condition'].nunique() == 55
        assert not predictions.duplicated(['model','condition','source_window_index']).any()
        numbers = predictions[['valence_true','arousal_true','valence_pred','arousal_pred']]
        assert np.isfinite(numbers.to_numpy()).all()
        # Models and conditions must retain exactly the same targets and windows.
        window_columns = ['subject_id','trial_id','start_s','valence_true','arousal_true']
        by_window = predictions.groupby('source_window_index')[window_columns].nunique()
        assert by_window.eq(1).all().all()
        assert predictions.groupby('source_window_index').size().eq(9*55).all()
        computed = []
        for (model, condition, subject), group in predictions.groupby(['model','condition','subject_id']):
            row = dict(model=model, condition=condition, subject_id=subject, n_windows=len(group))
            for target in ['valence','arousal']:
                actual = group[f'{target}_true'].to_numpy(dtype=np.float64)
                estimate = group[f'{target}_pred'].to_numpy(dtype=np.float64)
                row[f'ccc_{target}'] = concordance(actual, estimate)
                row[f'mae_{target}'] = float(np.mean(np.abs(actual-estimate)))
            row['mean_ccc'] = (row['ccc_valence']+row['ccc_arousal'])/2
            row['mean_mae'] = (row['mae_valence']+row['mae_arousal'])/2
            computed.append(row)
        key = ['model','condition','subject_id']
        comparison = pd.DataFrame(computed).merge(stored, on=key, validate='one_to_one',
                                                   suffixes=('_recomputed','_stored'))
        assert len(comparison) == 9*55*6
        assert comparison.n_windows_recomputed.eq(comparison.n_windows_stored).all()
        max_errors = {}
        for metric in METRICS:
            a = comparison[metric+'_recomputed'].to_numpy()
            b = comparison[metric+'_stored'].to_numpy()
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
            max_errors[metric] = float(np.max(np.abs(a-b)))
        reports.append(dict(fold=fold, prediction_rows=len(predictions),
                            participant_metric_rows=len(comparison),
                            prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            metrics_sha256=hashlib.sha256((REV/'results'/f'subject_metrics_{fold}.csv').read_bytes()).hexdigest(),
                            maximum_absolute_metric_differences=max_errors))
    report = dict(complete=len(files)==15, folds=len(files),
                  check='Independent native-scale CCC/MAE recomputation; paired windows/targets; finite predictions',
                  tolerances=dict(absolute=1e-6, relative=1e-6), reports=reports)
    destination = REV/'manifests'/('prediction_validation_partial.json' if args.allow_partial else 'prediction_validation.json')
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=report['complete'], folds=len(files),
                         prediction_rows=sum(r['prediction_rows'] for r in reports),
                         metric_rows=sum(r['participant_metric_rows'] for r in reports)), indent=2))


if __name__ == '__main__':
    main()
