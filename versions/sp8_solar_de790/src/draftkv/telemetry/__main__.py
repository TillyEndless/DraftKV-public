"""python -m draftkv.telemetry: JSONL -> JSON / CSV / Markdown, no CUDA."""
import argparse
import csv
import json
from pathlib import Path
import sys

from . import analyze, import_h200_csv, import_mac_nsight, read_jsonl, materialize_journal, EventWriter


def write_summary(summary, prefix):
    paths = [Path(str(prefix) + suffix) for suffix in ('.json', '.csv', '.md')]
    if any(p.exists() for p in paths):
        raise ValueError('output already exists; choose a fresh prefix')
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n'
    # Preflight before opening any files. All outputs use exclusive creation.
    with paths[0].open('x') as stream:
        stream.write(payload)
    samples = summary.get('samples', [])
    with paths[1].open('x', newline='') as stream:
        if 'samples' in summary:
            fields = ['run_id', 'sample_id', 'layer', 'rank', 'mode', 'valid', 'overlap_verified',
                      'interval_overlap_ms', 'active_math_overlap_ms', 'exposed_h2d_ms',
                      'consumer_ms', 'pipeline_critical_path_ms', 'frontier_waiting_fraction', 'errors', 'blockers']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in samples:
                values = {k: row[k] for k in fields[:5]}
                values.update({k: row['validation'][k] for k in ('valid', 'overlap_verified')})
                values.update({k: row['metrics'].get(k) for k in fields[7:13]})
                values['errors'] = '; '.join(row['validation']['errors'])
                values['blockers'] = '; '.join(row['validation']['blockers'])
                writer.writerow(values)
        else:
            writer = csv.DictWriter(stream, fieldnames=['scope', 'metric', 'value', 'unit', 'overlap_verified'])
            writer.writeheader()
            historical_rows = []
            for row in summary.get('original_rows', []):
                historical_rows += [(row['mode'], k, v) for k, v in row.items() if k != 'mode']
            for row in summary.get('ranges', []):
                historical_rows += [(row['range'], k, v) for k, v in row.items()
                                    if isinstance(v, (int, float))]
            for scope, metric, value in historical_rows:
                writer.writerow({'scope': scope, 'metric': metric, 'value': value,
                                 'unit': 'ms' if metric.endswith('_ms') else 'count',
                                 'overlap_verified': False})
    lines = ['# B01 overlap evidence', '', f"Kind: {summary['kind']}", '',
             f"overlap_verified: {str(summary['overlap_verified']).lower()}", '',
             'A: DMA/consumer interval intersection. B: observed MMA/DMA overlap. C: paired critical path reduction.', '',
             'A alone does not establish B or C. Release gates remain unassessed.', '']
    if samples:
        lines += ['| Scope | Metric | N | Median | P90 |', '|---|---|---:|---:|---:|']
        for scope, stats in [('global', summary['global'])] + [
                ('layer ' + k, v) for k, v in summary['by_layer'].items()]:
            for metric, dist in stats.items():
                lines.append(f"| {scope} | {metric} | {dist['count']} | {dist['median']} | {dist['p90']} |")
        lines += ['', '## Sample issues', '']
        for row in samples:
            text = '; '.join(row['validation']['errors'] + row['validation']['blockers'] + row['validation']['warnings'])
            # Escape embedded newlines in diagnostic strings.
            lines.append('- ' + json.dumps([row['run_id'], row['sample_id'], text], ensure_ascii=True))
        lines += ['', '## Paired savings', '', '```json', json.dumps(summary['single_host_sp_pairs'], indent=2), '```',
                  '', '## Observed pipeline spans', '', '```json',
                  json.dumps(summary['observed_pipeline_by_rank'], indent=2), '```']
    else:
        lines += ['Evidence levels: ' + ', '.join(summary['evidence_levels']), '', '```json',
                  json.dumps({k: v for k, v in summary.items() if k.startswith('historical_')}, indent=2), '```']
        for row in summary.get('ranges', []):
            lines += ['', '```json', json.dumps({k: v for k, v in row.items() if k != 'original'}, indent=2), '```']
    with paths[2].open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    analyze_parser = commands.add_parser('analyze')
    analyze_parser.add_argument('inputs', type=Path, nargs='+')
    analyze_parser.add_argument('--artifact-root', type=Path)
    analyze_parser.add_argument('--out', type=Path, required=True)
    finalize = commands.add_parser('finalize-journal')
    finalize.add_argument('input', type=Path)
    finalize.add_argument('--metadata', type=Path, required=True)
    finalize.add_argument('--out', type=Path, required=True)
    for name in ('import-h200', 'import-mac-nsight'):
        sub = commands.add_parser(name)
        sub.add_argument('input', type=Path)
        sub.add_argument('--source-revision', required=True)
        sub.add_argument('--expected-sha256', required=name == 'import-mac-nsight')
        sub.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'finalize-journal':
            records = materialize_journal(args.input, json.loads(args.metadata.read_text()))
            with EventWriter(args.out, run_id=records[0]['run_id']) as writer:
                for record in records:
                    writer.write(record)
            return 0
        if args.command == 'analyze':
            records = [r for p in args.inputs for r in read_jsonl(p)]
            if not records:
                raise ValueError('empty trace dataset')
            summary = analyze(records, artifact_root=args.artifact_root)
        else:
            importer = import_h200_csv if args.command == 'import-h200' else import_mac_nsight
            kwargs = {'source_revision': args.source_revision}
            if args.expected_sha256:
                kwargs['expected_sha256'] = args.expected_sha256
            summary = importer(args.input, **kwargs)
        write_summary(summary, args.out)
        if args.command == 'analyze' and any(not row['validation']['valid'] for row in summary['samples']):
            return 2
        # Exit zero means offline processing completed, never GPU acceptance.
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('telemetry: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
