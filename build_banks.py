"""Resumable FAISS banks with cached normalization and bounded query batches."""
import argparse
import hashlib
import json
import os
from pathlib import Path

# Avoid spawning more math-library workers than the job allocation permits.
THREADS = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, str(THREADS))
import numpy as np
import faiss


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path):
    with Path(path).open() as f:
        return json.load(f)


def unit_matrix(records):
    # Same per-vector normalization as core.pool, done just once.
    result = []
    for record in records:
        v = np.asarray(record['embedding'], dtype=np.float64)
        norm = np.linalg.norm(v)
        if not np.isfinite(v).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid embedding: ' + record['id'])
        result.append(v / norm)
    return np.stack(result)


def build(out, batch_size=128, stop_after_batches=None):
    import fcntl
    out = Path(out)
    with (out / 'bank_build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _build(out, batch_size, stop_after_batches)


def _build(out, batch_size, stop_after_batches):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    faiss.omp_set_num_threads(THREADS)
    checkpoint = out / 'bank_checkpoints'
    checkpoint.mkdir(exist_ok=True)
    fingerprint = dict(records_sha256=digest(out / 'records.json'),
                       builder_sha256=digest(__file__), batch_size=batch_size,
                       candidate_pool_limit=50, numpy=np.__version__, faiss=faiss.__version__)
    state = checkpoint / 'manifest.json'
    if state.exists() and load(state) != fingerprint:
        raise ValueError('Bank checkpoint inputs, code, batch size or library versions changed. '
                         'Use a new output directory or archive bank_checkpoints before rebuilding.')
    save_atomic(state, fingerprint)
    # The completion marker must not survive an incomplete rebuild.
    (out / 'bank_audit.json').unlink(missing_ok=True)
    rows = load(out / 'records.json')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate record IDs')
    pools = {}
    completed = 0
    for modality in ('ecg', 'cxr'):
        queries = [r for r in rows if r['modality'] == modality]
        bank = [r for r in queries if r['split'] == 'train']
        if len({r['patient_id'] for r in bank}) < 2:
            raise ValueError('Need two training patients for ' + modality)
        x = unit_matrix(bank)
        qmat = unit_matrix(queries)
        ids = np.asarray([r['id'] for r in bank])
        patients = np.asarray([r['patient_id'] for r in bank])
        x32 = np.asarray([r['embedding'] for r in bank], dtype='float32')
        faiss.normalize_L2(x32)
        index = faiss.IndexFlatIP(x32.shape[1])
        index.add(x32)
        temp_index = out / f'{modality}.faiss.tmp'
        faiss.write_index(index, str(temp_index))
        os.replace(temp_index, out / f'{modality}.faiss')
        save_atomic(out / f'{modality}_ids.json', ids.tolist())
        print(f'{modality}: {len(bank)} references, {len(queries)} queries; batch size {batch_size}', flush=True)
        for start in range(0, len(queries), batch_size):
            chunk = queries[start:start + batch_size]
            path = checkpoint / f'{modality}_{start:08d}.json'
            if path.exists():
                saved = load(path)
                payload = saved['pools']
                if (saved['sha256'] != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                        or set(payload) != {r['id'] for r in chunk}):
                    raise ValueError('Corrupted checkpoint: ' + str(path))
                pools.update(payload)
                print(f'{modality}: resumed {start + len(chunk)}/{len(queries)} queries', flush=True)
                continue
            q64 = qmat[start:start + len(chunk)]
            cosine = q64 @ x.T
            q32 = np.asarray([r['embedding'] for r in chunk], dtype='float32')
            faiss.normalize_L2(q32)
            scores, positions = index.search(q32, len(bank))
            reference_scores = np.take_along_axis(cosine, positions, axis=1)
            if not np.all(np.abs(scores - reference_scores) < 1e-5):
                raise ValueError('FAISS/cosine discrepancy in ' + modality)
            payload = {}
            for j, query in enumerate(chunk):
                eligible = np.flatnonzero(patients != query['patient_id'])
                count = min(50, len(eligible))
                threshold = np.partition(cosine[j, eligible], -count)[-count]
                # Recompute boundary candidates with the scalar dot used by core.pool.
                # Include near ties before the final score/ID sort.
                shortlist = eligible[cosine[j, eligible] >= threshold - 1e-12]
                scored = [(float(np.dot(q64[j], x[i])), str(ids[i])) for i in shortlist]
                scored.sort(key=lambda item: (-item[0], item[1]))
                payload[query['id']] = [rid for _, rid in scored[:count]]
            save_atomic(path, dict(pools=payload, sha256=hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()).hexdigest()))
            pools.update(payload)
            completed += 1
            print(f'{modality}: checkpoint saved {start + len(chunk)}/{len(queries)} queries', flush=True)
            if stop_after_batches is not None and completed >= stop_after_batches:
                return False
    if digest(out / 'records.json') != fingerprint['records_sha256']:
        raise ValueError('Input records changed during construction')
    if set(pools) != {r['id'] for r in rows}:
        raise ValueError('Query coverage mismatch')
    save_atomic(out / 'pools.json', pools)
    save_atomic(out / 'bank_audit.json', dict(queries=len(pools), training_only=True,
        own_patient_excluded=True, faiss_cosine_agreement=True, candidate_pool_limit=50,
        checkpointed=True, records_sha256=fingerprint['records_sha256']))
    print('READY: all banks and query pools completed.', flush=True)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    build(args.output_directory, args.batch_size)
