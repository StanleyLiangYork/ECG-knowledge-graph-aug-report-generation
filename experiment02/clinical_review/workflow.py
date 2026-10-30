"""Single-clinician assessment; no model inference or automatic clinical judgments.

Kept below a subdirectory to preserve BR00's signed root Python inventory.
"""
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import atomic, immutable, read, sha, signature, csv_save

RATING_FIELDS = ('reviewer_id', 'agreement', 'clinically_significant_error_count',
                 'clinically_significant_omission_count', 'minor_error_count',
                 'overall_quality_1_to_5', 'uninterpretable_image',
                 'discrepancy_details', 'comments')
COUNT_FIELDS = RATING_FIELDS[2:5]

INSTRUCTIONS = '''One ECG-qualified clinician performs the initial review and a separate
adjudication pass. Adjudication is a second assessment by the SAME clinician;
it is not independent confirmation or inter-rater reliability.

First interpret each query ECG without generated reports, retrieved examples,
automatic diagnoses, targets, model names or experimental arms. Submit those
interpretations before opening the randomized report assessment packet.
Judge each generated report against the query ECG and your independent interpretation.
Reference patients' findings are not facts about the query patient.

Major error: a clinically significant incorrect assertion potentially affecting
clinical interpretation or management. Major omission: a clinically significant
ECG finding missing from the report. Count minor errors separately. Prespecify
examples during calibration (rhythm, conduction, QT, ST-T, ectopy, voltage).
Do not assume an amplitude-normalized image preserves absolute voltage or that
image appearance supports a numeric QT interval. Record limitations and uncertainty.
Quality: 1 unusable; 2 major corrections; 3 moderate corrections; 4 minor
corrections; 5 acceptable unchanged. This is research assessment, not patient care.

Agreement: agree / disagree / uncertain. Record any clinical discrepancy and
its supporting ECG evidence. Null means UNASSESSED, never no errors.
Any disagreement, uncertainty, error, omission, quality <=3 or uninterpretable
image requires adjudication. Keep the original rating unchanged. Adjudication
may confirm the discrepancy, correct the initial rating, or remain unresolved.
Never revise a judgment simply to agree with a model or improve study results.
Specify the same source-access policy for all items. Model style and repeated
ECGs may partly reveal identity; avoid presenting paired reports side by side.
Only PUBLIC is for the clinician. PRIVATE reveals patient/model/arm linkage.
'''


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _ids(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate ' + key)
    return result


def _signed(folder, name):
    path = Path(folder) / name
    return dict(file=str(path), sha256=sha(path))


def _check(folder):
    folder = Path(folder)
    protocol = read(folder / 'PRIVATE/protocol.json')
    if protocol['helper_sha256'] != sha(__file__):
        raise ValueError('Review helper changed; preserve this version for this review')
    for item in protocol['sources']:
        if sha(item['file']) != item['sha256']:
            raise ValueError('Frozen review source changed: ' + item['file'])
    for item in read(folder / 'PRIVATE/assets.json'):
        if sha(folder / item['file']) != item['sha256']:
            raise ValueError('Review asset changed: ' + item['file'])
    released = folder / 'PUBLIC/reports.json'
    if released.exists() and read(released) != read(folder / 'PRIVATE/pending_reports.json'):
        raise ValueError('Released report text changed')
    return protocol


def prepare(cfg, folders, destination, n_queries=10, seed=43,
            arms=('bidirectional_pool', 'bidirectional_graph'), reviewer_id='',
            stage='calibration', sample_size_justification='', allow_partial=False,
            reviewer_expertise='',
            source_access_policy='Generator input ECG image only; record image limitations.'):
    """Create once. Resume clinical entry from per-item files; never overwrite a package."""
    from common import assert_config
    from evaluation import load_results
    from queries import load_queries, image_provider
    import shutil
    import tempfile

    assert_config(cfg)
    if not _text(reviewer_id) or not _text(source_access_policy) or not _text(reviewer_expertise):
        raise ValueError('Specify one reviewer identity, ECG expertise and a fixed source-access policy')
    if stage not in ('calibration', 'main') or not _text(sample_size_justification):
        raise ValueError('Specify stage and sample-size justification; 10 cases is calibration only')
    if type(n_queries) is not int or n_queries < 1 or not folders:
        raise ValueError('Positive sample size and at least one generator required')
    if len(set(arms)) != len(arms) or not {'bidirectional_pool', 'bidirectional_graph'} <= set(arms):
        raise ValueError('Include the unique primary paired arms')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Existing review preserved; reopen it or choose a new version')
    packet = Path(cfg['run_dir']) / 'packets' / cfg['cohort'] / 'manifest.json'
    manifest = read(packet)
    manifest['_file_sha256'] = sha(packet)
    if not set(arms) <= set(manifest['arms']):
        raise ValueError('Selected arm absent from frozen packets')
    data = {m: load_results(f, manifest) for m, f in folders.items()}
    coverage = {}
    for m, rows in data.items():
        coverage[m] = {}
        for a in arms:
            cases = [rows.get((q, a)) for q in manifest['query_ids']]
            coverage[m][a] = dict(expected=len(cases),
                ok=sum(r is not None and r['status'] == 'ok' for r in cases),
                failed=sum(r is not None and r['status'] != 'ok' for r in cases),
                not_attempted=sum(r is None for r in cases))
    if not allow_partial and any(v['failed'] or v['not_attempted'] for d in coverage.values() for v in d.values()):
        raise ValueError('Finish selected reports or prespecify allow_partial=True. Coverage: ' + str(coverage))
    shared = sorted(q for q in manifest['query_ids'] if all(
        rows.get((q, a), {}).get('status') == 'ok' for rows in data.values() for a in arms))
    queries = _ids(load_queries(cfg), 'id')
    rng = random.Random(seed)
    by_patient = defaultdict(list)
    for q in shared:
        by_patient[str(queries[q]['patient_id'])].append(q)
    patient_ids = sorted(by_patient)
    rng.shuffle(patient_ids)
    eligible_patients = len(patient_ids)
    if n_queries > eligible_patients:
        raise ValueError(f'Requested {n_queries} patients; only {eligible_patients} jointly complete unique patient IDs')
    selected = [rng.choice(sorted(by_patient[p])) for p in patient_ids[:n_queries]]
    source_files = [packet, Path(cfg['run_dir']) / 'protocol.json']
    for f in folders.values():
        source_files += [Path(f) / 'protocol.json', Path(f) / 'reports.jsonl']
    protocol = dict(version='single_reviewer_v1', cohort=cfg['cohort'], stage=stage,
        n_queries=n_queries, models=list(folders), arms=list(arms), seed=seed,
        reviewer_id=reviewer_id, reviewer_expertise=reviewer_expertise,
        sample_size_justification=sample_size_justification,
        source_access_policy=source_access_policy, allow_partial=allow_partial,
        sources=[dict(file=str(p.resolve()), sha256=sha(p)) for p in source_files],
        helper_sha256=sha(__file__),
        sampling='Seeded uniform patient-ID sample, one uniformly selected query per ID; jointly successful selected models/arms only',
        adjudication='Same reviewer, separate pass; retain, revise with reason, or unresolved',
        primary_endpoint='Report contains >=1 clinically significant error',
        primary_contrast='bidirectional_graph minus bidirectional_pool; lower is better',
        limitations=['Single reviewer; no inter-rater reliability',
            'Review conditional on successful generation; report failures separately',
            'Patient IDs may not prove distinct biological patients across external records',
            'Clinical review does not establish patient outcomes or causal reasoning'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix='.clinical_', dir=destination.parent))
    try:
        public = stage_dir / 'PUBLIC'
        (public / 'images').mkdir(parents=True)
        (stage_dir / 'PRIVATE').mkdir(mode=0o700)
        refs, entries, linkage, assets = [], [], [], []
        provider = image_provider(cfg)
        for i, q in enumerate(selected):
            cid = f'case_{i + 1:05d}'
            image, tensor = provider(queries[q])
            for rows in data.values():
                for a in arms:
                    r = rows[q, a]
                    if (sha(image), tensor) != (r['sensor_sha256'], r['source_tensor_sha256']):
                        raise ValueError('Generated report and reviewer ECG differ')
                    if str(r['patient_id']) != str(queries[q]['patient_id']):
                        raise ValueError('Report patient mismatch')
            relative = 'PUBLIC/images/' + cid + '.png'
            shutil.copyfile(image, stage_dir / relative)
            assets.append(dict(file=relative, sha256=sha(stage_dir / relative)))
            refs.append(dict(case_id=cid, image='images/' + cid + '.png',
                reviewer_id=reviewer_id, interpretable=None, independent_ecg_interpretation=None,
                limitations=None))
            for model in sorted(data):
                for arm in arms:
                    entries.append(dict(case_id=cid, image='images/' + cid + '.png',
                        generated_report=data[model][q, arm]['text']))
                    linkage.append(dict(query_id=q, patient_id=str(queries[q]['patient_id']),
                        model=model, arm=arm))
        order = list(range(len(entries)))
        rng.shuffle(order)
        blind, key = [], []
        for j, i in enumerate(order):
            rid = f'review_{j + 1:05d}'
            blind.append(dict(review_id=rid, **entries[i]))
            key.append(dict(review_id=rid, **linkage[i]))
        atomic(public / 'reference_cases.json', refs)
        atomic(stage_dir / 'PRIVATE/pending_reports.json', blind)
        atomic(stage_dir / 'PRIVATE/linkage.json', key)
        atomic(stage_dir / 'PRIVATE/assets.json', assets)
        atomic(stage_dir / 'PRIVATE/protocol.json', protocol)
        atomic(stage_dir / 'PRIVATE/coverage.json', dict(models=coverage,
            expected_queries=len(manifest['query_ids']), jointly_complete_queries=len(shared),
            eligible_patient_ids=eligible_patients, sampled_patient_ids=n_queries,
            excluded_queries=sorted(set(manifest['query_ids']) - set(shared))))
        (public / 'INSTRUCTIONS.txt').write_text(INSTRUCTIONS, encoding='utf-8')
        # Protect exact blank references, report text and linkage independently of later ratings.
        for name in ('PUBLIC/reference_cases.json', 'PRIVATE/pending_reports.json', 'PRIVATE/linkage.json'):
            assets.append(dict(file=name, sha256=sha(stage_dir / name)))
        atomic(stage_dir / 'PRIVATE/assets.json', assets)
        stage_dir.rename(destination)
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return dict(destination=str(destination), cases=n_queries, reports=len(entries),
        next='Clinician completes reference interpretations before BR08 releases reports')


def save_reference(folder, case_id, *, reviewer_id, interpretable,
                   independent_ecg_interpretation, limitations=''):
    protocol = _check(folder)
    if (reviewer_id != protocol['reviewer_id'] or type(interpretable) is not bool
            or not _text(independent_ecg_interpretation) or not isinstance(limitations, str)):
        raise ValueError('Complete the independent interpretation and boolean interpretability')
    if case_id not in _ids(read(Path(folder) / 'PUBLIC/reference_cases.json'), 'case_id'):
        raise ValueError('Unknown case')
    if (Path(folder) / 'PRIVATE/references_submitted.json').exists():
        raise ValueError('Independent references already frozen; do not rewrite after seeing reports')
    atomic(Path(folder) / 'PUBLIC/reference_drafts' / (case_id + '.json'),
        dict(case_id=case_id, reviewer_id=reviewer_id, interpretable=interpretable,
             independent_ecg_interpretation=independent_ecg_interpretation, limitations=limitations,
             saved_at_utc=datetime.now(timezone.utc).isoformat()))


def release_reports(folder):
    protocol = _check(folder)
    folder = Path(folder)
    refs = []
    for blank in read(folder / 'PUBLIC/reference_cases.json'):
        path = folder / 'PUBLIC/reference_drafts' / (blank['case_id'] + '.json')
        if not path.is_file():
            raise ValueError('Missing independent interpretation: ' + blank['case_id'])
        r = read(path)
        if (r['case_id'] != blank['case_id'] or r['reviewer_id'] != protocol['reviewer_id']
                or type(r['interpretable']) is not bool or not _text(r['independent_ecg_interpretation'])):
            raise ValueError('Invalid independent interpretation')
        refs.append(r)
    immutable(folder / 'PRIVATE/references_submitted.json', refs)
    immutable(folder / 'PUBLIC/reports.json', read(folder / 'PRIVATE/pending_reports.json'))
    immutable(folder / 'PUBLIC/reference_interpretations.json', refs)
    return dict(reports=len(read(folder / 'PUBLIC/reports.json')), next='Complete initial blinded ratings')


def validate_rating(r, reviewer):
    if set(r) != set(RATING_FIELDS) or r['reviewer_id'] != reviewer:
        raise ValueError('Rating schema/reviewer mismatch')
    if r['agreement'] not in ('agree', 'disagree', 'uncertain'):
        raise ValueError('Agreement must be agree, disagree or uncertain')
    if any(type(r[k]) is not int or r[k] < 0 for k in COUNT_FIELDS):
        raise ValueError('Error/omission counts must be explicit nonnegative integers')
    if type(r['overall_quality_1_to_5']) is not int or not 1 <= r['overall_quality_1_to_5'] <= 5:
        raise ValueError('Quality must be 1–5')
    if type(r['uninterpretable_image']) is not bool:
        raise ValueError('Interpretability must be boolean')
    if not all(isinstance(r[k], str) for k in ('discrepancy_details', 'comments')):
        raise ValueError('Details/comments must be strings')
    if r['agreement'] == 'agree' and any(r[k] for k in COUNT_FIELDS):
        raise ValueError('Nonzero error/omission count conflicts with agreement=agree')
    if needs_adjudication(r) and not _text(r['discrepancy_details']):
        raise ValueError('Explain the discrepancy, uncertainty or interpretability limitation')
    return r


def needs_adjudication(r):
    return (r['agreement'] != 'agree' or any(r[k] > 0 for k in COUNT_FIELDS)
            or r['overall_quality_1_to_5'] <= 3 or r['uninterpretable_image'])


def save_rating(folder, review_id, **rating):
    protocol = _check(folder)
    folder = Path(folder)
    if not (folder / 'PRIVATE/references_submitted.json').is_file():
        raise ValueError('Freeze independent references before report review')
    reports = _ids(read(folder / 'PUBLIC/reports.json'), 'review_id')
    if review_id not in reports:
        raise ValueError('Unknown report')
    if (folder / 'PRIVATE/initial_submitted.json').exists():
        raise ValueError('Initial ratings frozen; use the adjudication step')
    validate_rating(rating, protocol['reviewer_id'])
    atomic(folder / 'PUBLIC/rating_drafts' / (review_id + '.json'),
        dict(review_id=review_id, saved_at_utc=datetime.now(timezone.utc).isoformat(), **rating))


def _initial(folder, protocol):
    rows = read(Path(folder) / 'PRIVATE/initial_submitted.json')
    for r in rows:
        validate_rating({k: r[k] for k in RATING_FIELDS}, protocol['reviewer_id'])
    return rows


def prepare_adjudication(folder):
    protocol = _check(folder)
    folder = Path(folder)
    reports = _ids(read(folder / 'PUBLIC/reports.json'), 'review_id')
    frozen = folder / 'PRIVATE/initial_submitted.json'
    if frozen.exists():
        ratings = _initial(folder, protocol)
    else:
        ratings = []
        for rid in reports:
            path = folder / 'PUBLIC/rating_drafts' / (rid + '.json')
            if not path.is_file():
                raise ValueError('Unassessed report: ' + rid)
            r = read(path)
            if r.get('review_id') != rid:
                raise ValueError('Rating ID mismatch')
            validate_rating({k: r[k] for k in RATING_FIELDS}, protocol['reviewer_id'])
            ratings.append(r)
        immutable(frozen, ratings)
    refs = _ids(read(folder / 'PRIVATE/references_submitted.json'), 'case_id')
    queue = []
    for r in ratings:
        item = reports[r['review_id']]
        if needs_adjudication(r) or not refs[item['case_id']]['interpretable']:
            queue.append(dict(**item, initial_rating=r,
                independent_interpretation=refs[item['case_id']],
                instruction='Recheck the query ECG. Retain disagreement if supported; do not force model agreement.'))
    immutable(folder / 'PUBLIC/adjudication_queue.json', queue)
    immutable(folder / 'PRIVATE/adjudication_protocol.json', dict(
        initial_sha256=sha(frozen), references_sha256=sha(folder / 'PRIVATE/references_submitted.json'),
        queue_sha256=sha(folder / 'PUBLIC/adjudication_queue.json'), reviewer_id=protocol['reviewer_id']))
    return dict(initial_ratings=len(ratings), adjudications_required=len(queue),
                next='Same clinician completes a separate adjudication pass, preferably in a later session')


def save_adjudication(folder, review_id, *, reviewer_id, decision, reason,
                      evidence_reviewed, final_rating=None):
    protocol = _check(folder)
    folder = Path(folder)
    queue = _ids(read(folder / 'PUBLIC/adjudication_queue.json'), 'review_id')
    if review_id not in queue or reviewer_id != protocol['reviewer_id']:
        raise ValueError('Unknown adjudication item or different reviewer')
    if decision not in ('retain_initial', 'revise_initial', 'unresolved'):
        raise ValueError('Invalid decision')
    if not _text(reason) or not _text(evidence_reviewed):
        raise ValueError('Record clinical rationale and the ECG evidence actually rechecked')
    initial = queue[review_id]['initial_rating']
    original = {k: initial[k] for k in RATING_FIELDS}
    if decision == 'retain_initial':
        if final_rating is not None and final_rating != original:
            raise ValueError('Retain decision cannot alter initial rating')
        final_rating = original
    elif decision == 'revise_initial':
        validate_rating(final_rating or {}, reviewer_id)
        if final_rating == original:
            raise ValueError('Revision must change a rating; otherwise retain initial')
    elif final_rating is not None:
        raise ValueError('Unresolved item must not silently receive a final clinical score')
    record = dict(review_id=review_id, reviewer_id=reviewer_id, decision=decision,
        reason=reason, evidence_reviewed=evidence_reviewed, final_rating=final_rating,
        initial_rating_sha256=signature(initial))
    # A submitted adjudication is immutable. Correct errors in a new review version.
    path = folder / 'PUBLIC/adjudications' / (review_id + '.json')
    record['submitted_at_utc'] = read(path)['submitted_at_utc'] if path.exists() else datetime.now(timezone.utc).isoformat()
    immutable(path, record)


def progress(folder):
    folder = Path(folder)
    refs = read(folder / 'PUBLIC/reference_cases.json')
    reports = read(folder / 'PUBLIC/reports.json') if (folder / 'PUBLIC/reports.json').exists() else []
    queue = read(folder / 'PUBLIC/adjudication_queue.json') if (folder / 'PUBLIC/adjudication_queue.json').exists() else []
    return dict(reference_cases=len(refs), references_saved=sum((folder / 'PUBLIC/reference_drafts' / (r['case_id'] + '.json')).exists() for r in refs),
        reports_released=len(reports), initial_ratings_saved=sum((folder / 'PUBLIC/rating_drafts' / (r['review_id'] + '.json')).exists() for r in reports),
        adjudications_required=len(queue), adjudications_saved=sum((folder / 'PUBLIC/adjudications' / (r['review_id'] + '.json')).exists() for r in queue))


def analyze(folder, destination, repeats=2000, seed=43):
    """Require every initial rating and every triggered adjudication; report unresolved explicitly."""
    import numpy as np
    protocol = _check(folder)
    folder, destination = Path(folder), Path(destination)
    if type(repeats) is not int or repeats < 100:
        raise ValueError('At least 100 patient-bootstrap replicates required')
    initial = _initial(folder, protocol)
    reports = _ids(read(folder / 'PRIVATE/pending_reports.json'), 'review_id')
    keys = _ids(read(folder / 'PRIVATE/linkage.json'), 'review_id')
    if set(reports) != set(_ids(initial, 'review_id')):
        raise ValueError('Initial review incomplete')
    ap = read(folder / 'PRIVATE/adjudication_protocol.json')
    if (ap['initial_sha256'] != sha(folder / 'PRIVATE/initial_submitted.json')
            or ap['references_sha256'] != sha(folder / 'PRIVATE/references_submitted.json')
            or ap['queue_sha256'] != sha(folder / 'PUBLIC/adjudication_queue.json')):
        raise ValueError('Submitted references, initial ratings or adjudication queue changed')
    refs = _ids(read(folder / 'PRIVATE/references_submitted.json'), 'case_id')
    rows, decisions, sources = [], Counter(), []
    for r in initial:
        rid = r['review_id']
        flagged = needs_adjudication(r) or not refs[reports[rid]['case_id']]['interpretable']
        final, decision = r, 'not_required'
        if flagged:
            path = folder / 'PUBLIC/adjudications' / (rid + '.json')
            if not path.exists():
                raise ValueError('Adjudication not completed: ' + rid)
            adjudication = read(path)
            sources.append(_signed(path.parent, path.name))
            if adjudication['initial_rating_sha256'] != signature(r):
                raise ValueError('Adjudication does not match original assessment')
            decision, final = adjudication['decision'], adjudication['final_rating']
            if (decision not in ('retain_initial', 'revise_initial', 'unresolved')
                    or adjudication['reviewer_id'] != protocol['reviewer_id']
                    or not _text(adjudication['reason']) or not _text(adjudication['evidence_reviewed'])):
                raise ValueError('Invalid adjudication')
            if decision == 'retain_initial' and final != {k: r[k] for k in RATING_FIELDS}:
                raise ValueError('Retained adjudication changed the initial assessment')
            if decision == 'unresolved' and final is not None:
                raise ValueError('Unresolved item has a clinical score')
            if final is not None:
                validate_rating(final, protocol['reviewer_id'])
        decisions[decision] += 1
        base = dict(keys[rid], case_id=reports[rid]['case_id'])
        base.pop('review_id', None)
        for phase, score in (('initial', r), ('adjudicated', final)):
            # Independently uninterpretable ECGs cannot support a definitive error rate.
            evaluable = (score is not None and score['agreement'] != 'uncertain'
                         and not score['uninterpretable_image'] and refs[reports[rid]['case_id']]['interpretable'])
            rows.append(dict(review_id=rid, **base, phase=phase, adjudication_decision=decision,
                resolved=score is not None, clinically_evaluable=evaluable,
                agreement=score['agreement'] if score is not None else None,
                clinically_significant_error_count=score['clinically_significant_error_count'] if score is not None else None,
                clinically_significant_omission_count=score['clinically_significant_omission_count'] if score is not None else None,
                minor_error_count=score['minor_error_count'] if score is not None else None,
                major_error=int(score['clinically_significant_error_count'] > 0) if evaluable else None,
                major_omission=int(score['clinically_significant_omission_count'] > 0) if evaluable else None,
                acceptable_unchanged=int(score['overall_quality_1_to_5'] == 5) if evaluable else None,
                quality=score['overall_quality_1_to_5'] if evaluable else None))
    summary, comparisons = [], []
    for phase in ('initial', 'adjudicated'):
        for model in protocol['models']:
            for arm in protocol['arms']:
                group = [r for r in rows if (r['phase'], r['model'], r['arm']) == (phase, model, arm)]
                known = [r for r in group if r['clinically_evaluable']]
                summary.append(dict(phase=phase, model=model, arm=arm, sampled=len(group),
                    evaluable=len(known), unresolved=sum(not r['resolved'] for r in group),
                    uncertain=sum(r['agreement'] == 'uncertain' for r in group),
                    unevaluable=len(group)-len(known),
                    **{metric: float(np.mean([r[metric] for r in known])) if known else None
                       for metric in ('major_error', 'major_omission', 'acceptable_unchanged', 'quality')},
                    major_error_lower_bound=sum(r['major_error'] for r in known)/len(group),
                    major_error_upper_bound=(sum(r['major_error'] for r in known)+len(group)-len(known))/len(group)))
            for metric in ('major_error', 'major_omission', 'acceptable_unchanged', 'quality'):
                lookup = {(r['query_id'], r['arm']): r for r in rows if r['phase'] == phase and r['model'] == model}
                patients = defaultdict(list)
                for (q, a), left in lookup.items():
                    right = lookup.get((q, 'bidirectional_pool'))
                    if a == 'bidirectional_graph' and right and left[metric] is not None and right[metric] is not None:
                        patients[left['patient_id']].append(left[metric] - right[metric])
                values = np.array([np.mean(v) for _, v in sorted(patients.items())])
                rng = np.random.default_rng(seed)
                boot = [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)] if len(values) else []
                ci = np.quantile(boot, [.025, .975]).tolist() if boot else [None, None]
                comparisons.append(dict(phase=phase, model=model, metric=metric,
                    contrast='bidirectional_graph_minus_bidirectional_pool',
                    estimate=float(values.mean()) if len(values) else None,
                    ci95_lower=ci[0], ci95_upper=ci[1], paired_patients=len(values),
                    paired_queries=sum(map(len, patients.values())),
                    interpretation='Negative favors graph for errors/omissions; positive for quality/acceptability'))
    destination.mkdir(parents=True, exist_ok=True)
    immutable(destination / 'protocol.json', dict(review_protocol=sha(folder / 'PRIVATE/protocol.json'),
        initial=sha(folder / 'PRIVATE/initial_submitted.json'), adjudications=sources,
        references=ap['references_sha256'], helper=sha(__file__), repeats=repeats, seed=seed))
    csv_save(destination / 'per_report.csv', rows)
    csv_save(destination / 'summary.csv', summary)
    csv_save(destination / 'paired_comparisons.csv', comparisons)
    result = dict(complete=True, stage=protocol['stage'], cohort=protocol['cohort'],
        reviewer_count=1, adjudication_decisions=dict(decisions), summary=summary, comparisons=comparisons,
        coverage=read(folder / 'PRIVATE/coverage.json'),
        limitations=protocol['limitations'] + ['Intervals measure case sampling, not variability across clinicians',
            'No inter-rater agreement statistic; same-reviewer adjudication is not independent validation',
            'Unresolved/uninterpretable cases excluded from paired rates; bounds and denominators reported',
            'Calibration/previously inspected test cohorts are exploratory; no automatic efficacy claim'])
    atomic(destination / 'results.json', result)
    (destination / 'findings.txt').write_text(
        'Single-clinician ECG report assessment with same-clinician adjudication.\n'
        f"Cohort: {protocol['cohort']}; stage: {protocol['stage']}.\n"
        f'Adjudication decisions: {dict(decisions)}.\n'
        'Primary endpoint: report contains at least one clinically significant error.\n'
        'Primary contrast: graph minus bidirectional pool; negative favors graph.\n'
        'Read paired_comparisons.csv for case-sampling confidence intervals.\n'
        'Initial and adjudicated assessments both retained. No inter-rater reliability is established.\n'
        'All results remain conditional on sampled successfully generated reports; see coverage.\n', encoding='utf-8')
    return result
