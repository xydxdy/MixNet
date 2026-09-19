"""
=============================================================================
 HOMEWORK PART 3 -- BENCHMARK YOUR PIPELINE AGAINST MixNet
=============================================================================

Collects the per-fold result CSVs that every `run_*.py` writes, aggregates
them the way the MixNet paper does (average the folds within a subject, then
report mean +/- std across subjects) and runs a paired significance test
between two runs.

This script is fully implemented -- you do not have to edit it. Your job is to
run it and to interpret the numbers in your report.

Usage
    # discover every run under logs/ and compare them
    python benchmark.py

    # compare two specific runs and name them
    python benchmark.py \
        --runs HWNet=logs/HWNet/subject_dependent_2_classes_BCIC2a \
               MixNet=logs/MixNet/subject_dependent_2_classes_BCIC2a_HistoricalTangentSlope_csp_components_2_margin_1.0_latent_dim_18_warmup_7

    # pick which two runs the paired test compares
    python benchmark.py --baseline MixNet --candidate HWNet

Outputs
    a per-subject table and a summary table on stdout (markdown, paste-ready)
    benchmark_per_subject.csv
    benchmark_summary.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# Columns produced by `BaseModel.evaluate()`. 'test_acc' is the accuracy on the
# held-out session E; 'f1-score' is macro/binary depending on num_class.
METRICS = ['test_acc', 'f1-score']
RESULT_GLOB = 'S*_all_results.csv'
SUBJECT_RE = re.compile(r'S(\d+)_all_results\.csv$')


def discover_runs(log_dir):
    """Find every directory under `log_dir` that holds per-subject results.

    Runs are named after their top-level folder (the model name) when that is
    unambiguous, so the tables stay readable; colliding runs keep their full
    relative path to stay distinguishable.
    """
    found = []
    for dirpath, _dirnames, filenames in os.walk(log_dir):
        if any(SUBJECT_RE.search(f) for f in filenames):
            rel = os.path.relpath(dirpath, log_dir).replace(os.sep, '/')
            found.append((rel, dirpath))

    short_counts = {}
    for rel, _ in found:
        short = rel.split('/')[0]
        short_counts[short] = short_counts.get(short, 0) + 1

    runs = {}
    for rel, dirpath in found:
        short = rel.split('/')[0]
        runs[short if short_counts[short] == 1 else rel] = dirpath
    return dict(sorted(runs.items()))


def parse_runs(run_args, log_dir):
    """Turn `NAME=PATH` arguments into a {name: path} mapping."""
    if not run_args:
        return discover_runs(log_dir)
    runs = {}
    for item in run_args:
        if '=' in item:
            name, path = item.split('=', 1)
        else:
            name, path = os.path.basename(item.rstrip('/')), item
        runs[name] = path
    return runs


def load_run(name, run_dir):
    """Read one run into a tidy frame: one row per (subject, fold)."""
    files = sorted(glob.glob(os.path.join(run_dir, RESULT_GLOB)))
    if not files:
        print('  [skip] {}: no {} found in {}'.format(name, RESULT_GLOB, run_dir))
        return None

    frames = []
    for path in files:
        match = SUBJECT_RE.search(os.path.basename(path))
        if match is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print('  [warn] {}: cannot read {} ({})'.format(name, path, exc))
            continue
        if df.empty:
            print('  [warn] {}: {} is empty, the run may have crashed'
                  .format(name, os.path.basename(path)))
            continue
        df = df.copy()
        df['subject'] = int(match.group(1))
        df['fold'] = np.arange(1, len(df) + 1)
        df['run'] = name
        frames.append(df)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    available = [m for m in METRICS if m in out.columns]
    missing = [m for m in METRICS if m not in out.columns]
    if missing:
        print('  [warn] {}: missing column(s) {} -- available: {}'
              .format(name, missing, list(out.columns)))
    if not available:
        return None

    n_folds = out.groupby('subject')['fold'].max()
    incomplete = n_folds[n_folds < n_folds.max()]
    if len(incomplete):
        print('  [warn] {}: incomplete subject(s) {} -- fewer folds than the '
              'rest, the comparison may be unfair'
              .format(name, list(incomplete.index)))

    print('  [ok]   {}: {} subjects x up to {} folds  ({})'
          .format(name, out['subject'].nunique(), int(n_folds.max()), run_dir))
    return out[['run', 'subject', 'fold'] + available]


def per_subject_table(raw):
    """Average the folds within each subject -- the unit of comparison."""
    metrics = [m for m in METRICS if m in raw.columns]
    return (raw.groupby(['run', 'subject'])[metrics]
               .mean()
               .reset_index())


def summary_table(per_subject):
    """Mean +/- std across subjects, per run and metric."""
    metrics = [m for m in METRICS if m in per_subject.columns]
    rows = []
    for run, group in per_subject.groupby('run', sort=False):
        row = {'run': run, 'n_subjects': len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[metric + '_mean'] = values.mean()
            row[metric + '_std'] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(per_subject, baseline, candidate, metric):
    """Paired comparison over the subjects shared by both runs."""
    try:
        from scipy import stats
    except ImportError:
        print('scipy is not installed, skipping the significance test.')
        return

    a = per_subject[per_subject['run'] == candidate].set_index('subject')[metric]
    b = per_subject[per_subject['run'] == baseline].set_index('subject')[metric]
    shared = sorted(set(a.index) & set(b.index))
    if len(shared) < 2:
        print('Not enough shared subjects between "{}" and "{}" to run a '
              'paired test.'.format(candidate, baseline))
        return

    a, b = a.loc[shared].to_numpy(float), b.loc[shared].to_numpy(float)
    diff = a - b

    print('\n### Paired comparison on {} ({} subjects)'.format(metric, len(shared)))
    print('{:<28}: {:.4f}'.format(candidate, a.mean()))
    print('{:<28}: {:.4f}'.format(baseline, b.mean()))
    print('{:<28}: {:+.4f}'.format('difference (cand - base)', diff.mean()))
    print('{:<28}: {}/{} subjects'.format('candidate wins on', int((diff > 0).sum()),
                                          len(shared)))

    t_stat, t_p = stats.ttest_rel(a, b)
    print('{:<28}: t = {:.4f}, p = {:.4f}'.format('paired t-test', t_stat, t_p))

    if np.allclose(diff, 0):
        print('{:<28}: skipped (identical results)'.format('Wilcoxon'))
    else:
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
            print('{:<28}: W = {:.4f}, p = {:.4f}'.format('Wilcoxon signed-rank',
                                                          w_stat, w_p))
        except ValueError as exc:                    # noqa: BLE001
            print('{:<28}: skipped ({})'.format('Wilcoxon', exc))


def to_markdown(df, float_fmt='{:.4f}'):
    """Render a frame as a markdown table without requiring the `tabulate` extra."""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: float_fmt.format(v))
        else:
            display[col] = display[col].astype(str)

    headers = list(display.columns)
    widths = [max(len(h), *(len(v) for v in display[h])) if len(display) else len(h)
              for h in headers]
    lines = ['| ' + ' | '.join(h.ljust(w) for h, w in zip(headers, widths)) + ' |',
             '|-' + '-|-'.join('-' * w for w in widths) + '-|']
    for _, row in display.iterrows():
        lines.append('| ' + ' | '.join(str(row[h]).ljust(w)
                                       for h, w in zip(headers, widths)) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='root searched when --runs is not given')
    parser.add_argument('--runs', nargs='+', default=None,
                        help='NAME=PATH pairs; default = discover under --log_dir')
    parser.add_argument('--baseline', type=str, default=None,
                        help='run name used as the baseline of the paired test')
    parser.add_argument('--candidate', type=str, default=None,
                        help='run name compared against the baseline')
    parser.add_argument('--out_prefix', type=str, default='benchmark',
                        help='prefix of the written CSV files')
    args = parser.parse_args()

    runs = parse_runs(args.runs, args.log_dir)
    if not runs:
        raise SystemExit(
            'No runs found under "{}". Train at least one model first, e.g.\n'
            '  python run_HWNet.py\n'
            '  python run_MixNet.py --dataset BCIC2a --train_type subject_dependent '
            "--data_type spectral_spatial_signals --n_component 2 --warmup 7"
            .format(args.log_dir))

    print('Collecting results from {} run(s):'.format(len(runs)))
    frames = [f for f in (load_run(n, p) for n, p in runs.items()) if f is not None]
    if not frames:
        raise SystemExit('None of the discovered directories contained usable results.')

    raw = pd.concat(frames, ignore_index=True)
    per_subject = per_subject_table(raw)
    summary = summary_table(per_subject)

    metrics = [m for m in METRICS if m in per_subject.columns]
    wide = per_subject.pivot(index='subject', columns='run', values=metrics)
    wide.columns = ['{} [{}]'.format(run, metric) for metric, run in wide.columns]
    wide = wide.reset_index()

    print('\n## Per-subject results (mean over folds)\n')
    print(to_markdown(wide))
    print('\n## Summary across subjects (mean +/- std)\n')
    print(to_markdown(summary))

    per_subject_csv = args.out_prefix + '_per_subject.csv'
    summary_csv = args.out_prefix + '_summary.csv'
    wide.to_csv(per_subject_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print('\nWritten: {} and {}'.format(per_subject_csv, summary_csv))

    names = list(summary['run'])
    baseline = args.baseline
    candidate = args.candidate
    if baseline is None or candidate is None:
        # Sensible default: compare MixNet against whichever other run is found.
        guess_base = next((n for n in names if 'MixNet' in n), None)
        guess_cand = next((n for n in names if n != guess_base), None)
        baseline = baseline or guess_base
        candidate = candidate or guess_cand

    if baseline and candidate and baseline != candidate:
        for metric in metrics:
            paired_test(per_subject, baseline, candidate, metric)
    elif len(names) >= 2:
        print('\nTwo or more runs found but no pair selected. Re-run with '
              '--baseline and --candidate, e.g.\n'
              '  python benchmark.py --baseline "{}" --candidate "{}"'
              .format(names[0], names[1]))


if __name__ == '__main__':
    main()
