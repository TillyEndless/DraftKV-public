"""Exclusive JSONL writer and strict reader, with no implicit event collection."""
import json
from pathlib import Path

from .schema import structural_errors


def _constant(value):
    raise ValueError('non-finite JSON number: ' + value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_constant, object_pairs_hook=_object)
                if not isinstance(record, dict):
                    raise ValueError('JSONL record must be an object')
                yield record
            except (ValueError, TypeError) as exc:
                raise ValueError(f'JSONL line {line_number}: {exc}') from exc


class EventWriter:
    """One fresh capture per file. Existing captures are never truncated/appended.

    One caller owns this writer; rank writers use separate files. Each successful
    write flushes a complete line. A read/analysis step still must check evidence.
    """
    def __init__(self, path, *, run_id):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('run_id required')
        self.run_id = run_id
        self.seen = set()
        self.capture = None
        self.stream = Path(path).open('x', encoding='utf-8')

    def write(self, record):
        errors = structural_errors(record)
        if errors:
            raise ValueError('; '.join(errors))
        if record['run_id'] != self.run_id:
            raise ValueError('cross-run write')
        if record['sample_id'] in self.seen:
            raise ValueError('sample already written')
        capture = (record['trace_id'], record['trace_sha256'])
        if self.capture is not None and capture != self.capture:
            raise ValueError('one writer requires one capture')
        line = json.dumps(record, allow_nan=False, sort_keys=True, separators=(',', ':'))
        self.stream.write(line + '\n')
        self.stream.flush()
        self.capture = capture
        self.seen.add(record['sample_id'])

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ObservationWriter:
    """Live event journal, before a profiler capture has a final SHA-256.

    Values are observations supplied by the caller. CUDA values must be resolved
    by the caller when ready; this API never synchronizes streams or reads clocks.
    Journals are pending observations, not verifiable sample records.
    """
    def __init__(self, path, *, run_id, flush_every=1):
        import atexit
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('run_id required')
        if type(flush_every) is not int or flush_every < 1:
            raise ValueError('flush_every must be a positive integer')
        self.run_id = run_id
        self.sequence = 0
        self.seen = set()
        self.flush_every = flush_every
        self._since_flush = 0
        self._closed = False
        self.stream = Path(path).open('x', encoding='utf-8', buffering=1024 * 1024)
        atexit.register(self.close)

    def mark(self, sample_id, event, value, *, clock_domain, event_unit):
        from .schema import STAMPS, UNITS, number
        if (not isinstance(sample_id, str) or not sample_id or event not in STAMPS
                or not isinstance(clock_domain, str) or not clock_domain
                or not isinstance(event_unit, str) or event_unit not in UNITS):
            raise ValueError('invalid live observation metadata')
        number(value)
        key = (sample_id, event)
        if key in self.seen:
            raise ValueError('observation already recorded')
        observation = {'schema_version': 1, 'kind': 'pending_observation', 'run_id': self.run_id,
                       'sample_id': sample_id, 'sequence': self.sequence, 'event': event,
                       'timestamp': {'value': value, 'clock_domain': clock_domain, 'event_unit': event_unit}}
        line = json.dumps(observation, allow_nan=False, sort_keys=True)
        if self._closed:
            raise ValueError('observation writer is closed')
        self.stream.write(line + '\n')
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.stream.flush()
            self._since_flush = 0
        self.sequence += 1
        self.seen.add(key)

    def flush(self):
        if self._closed:
            raise ValueError('observation writer is closed')
        self.stream.flush()
        self._since_flush = 0

    def close(self):
        if self._closed:
            return
        self.flush()
        self.stream.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def materialize_journal(path, metadata_by_sample):
    """Join live observations with immutable capture hashes after profiler export.

    A missing/duplicate event, sequence gap, run mismatch or missing metadata is a
    hard error. Nothing is inferred and pending records cannot verify overlap.
    """
    from copy import deepcopy
    from .schema import STAMPS
    if (not isinstance(metadata_by_sample, dict) or not metadata_by_sample
            or any(not isinstance(v, dict) for v in metadata_by_sample.values())):
        raise ValueError('sample metadata mapping required')
    records = {}
    run_id = None
    for sequence, row in enumerate(read_jsonl(path)):
        if (type(row.get('schema_version')) is not int or row['schema_version'] != 1
                or row.get('kind') != 'pending_observation'
                or type(row.get('sequence')) is not int or row['sequence'] != sequence):
            raise ValueError('invalid pending observation schema/sequence')
        if run_id is None:
            run_id = row.get('run_id')
        if not run_id or row.get('run_id') != run_id:
            raise ValueError('cross-run observation journal')
        sample_id, event = row.get('sample_id'), row.get('event')
        if not isinstance(sample_id, str) or sample_id not in metadata_by_sample or event not in STAMPS:
            raise ValueError('missing sample metadata or unknown event')
        if sample_id not in records:
            records[sample_id] = deepcopy(metadata_by_sample[sample_id])
            if records[sample_id].get('run_id') != run_id or records[sample_id].get('sample_id') != sample_id:
                raise ValueError('journal/metadata identity mismatch')
        record = records[sample_id]
        if event in record:
            raise ValueError('duplicate event or metadata overwrites observation')
        stamp = row.get('timestamp')
        if not isinstance(stamp, dict) or set(stamp) != {'value', 'clock_domain', 'event_unit'}:
            raise ValueError('journal timestamp requires explicit domain and unit')
        record[event] = stamp
    if set(records) != set(metadata_by_sample):
        raise ValueError('journal missing declared samples')
    for record in records.values():
        errors = structural_errors(record)
        if errors:
            raise ValueError('; '.join(errors))
    if len({(r['trace_id'], r['trace_sha256']) for r in records.values()}) != 1:
        raise ValueError('one journal requires one capture')
    return list(records.values())
