"""Reconcile completed human scope, frozen outputs, and reported paired effects."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'ball_quality_20261109/training_phase_scope_review'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    report = json.loads((OUT / 'evaluation_report.json').read_text(encoding='utf-8'))
    for table in [manifest['protected_input_hashes'], manifest['immutable_artifact_hashes']]:
        assert all(sha(Path(p)) == digest for p, digest in table.items())
    tracked = [OUT / name for name in ['phase_labels.csv', 'annotation_events.jsonl']]
    before = {str(p): sha(p) for p in tracked}
    assert before[str(tracked[0])] == report['phase_labels_sha256']
    assert before[str(tracked[1])] == report['events_sha256']
    phase_rows = read(tracked[0])
    phases = {r['sample_id']: r for r in phase_rows}
    initial = {r['sample_id']: r for r in read(OUT / 'initial_phase_labels.csv')}
    assert len(phases) == len(phase_rows) == 123
    replay = dict(initial)
    events = [json.loads(line) for line in tracked[1].read_text(encoding='utf-8').splitlines() if line]
    for event in events:
        assert len(event['before']) == len(event['after'])
        for old, new in zip(event['before'], event['after'], strict=True):
            sid = old['sample_id']
            assert sid == new['sample_id'] and replay[sid] == old and old['locked'] == '0'
            replay[sid] = new
    assert replay == phases
    assert all(r['reviewer_id'] == 'human_user' and r['review_status'] != 'unreviewed' for r in phase_rows)
    assert all(r == initial[r['sample_id']] for r in phase_rows if r['locked'] == '1')
    assert dict(Counter(r['phase_status'] for r in phase_rows)) == report['phase_counts']
    gt = {r['sample_id']: r for r in read(OUT / 'frozen_ball_groundtruth.csv')}
    per_frame = {(r['method'], r['sample_id']): r for r in read(OUT / 'phase_conditional_per_frame.csv')}
    pairs = {(r['method'], r['sample_id']): r for r in read(OUT / 'paired_changes.csv')}
    names = [m['method'] for m in manifest['frozen_methods']]
    assert len(per_frame) == len(pairs) == 4 * 123
    outcomes, predictions = {}, {}
    for method in manifest['frozen_methods']:
        name = method['method']
        path = Path(method['path'])
        assert sha(path) == method['sha256']
        ps = read(path)
        assert len(ps) == 123 and {r['sample_id'] for r in ps} == set(gt) == set(phases)
        predictions[name] = {r['sample_id']: r for r in ps}
        outcomes[name] = {}
        for r in ps:
            sid = r['sample_id']
            p, g = phases[sid], gt[sid]
            assert all(r[k] == g[k] for k in ['local_frame', 'source_frame', 'truth_visibility', 'truth_x_px', 'truth_y_px'])
            if g['truth_visibility'] == 'visible':
                if int(r['visible']):
                    d = math.dist((float(r['x_px']), float(r['y_px'])), (float(g['truth_x_px']), float(g['truth_y_px'])))
                    outcome = 'correct16px' if d <= 16 else 'wrong_position'
                else:
                    outcome = 'missing_visible_ball'
            else:
                outcome = 'output_when_not_observable' if int(r['visible']) else 'no_output_when_not_observable'
            outcomes[name][sid] = outcome
            expected = {'nonplay': 'excluded_nonplay', 'uncertain': 'excluded_phase_uncertain'}.get(p['phase_status'], outcome)
            assert per_frame[name, sid]['outcome'] == expected
            assert per_frame[name, sid]['phase_status'] == p['phase_status']
        counts = Counter(outcomes[name][sid] for sid in phases if phases[sid]['phase_status'] == 'play')
        totals = next(t for t in report['phase_conditional_totals'] if t['method'] == name)
        for key in ['correct16px', 'wrong_position', 'missing_visible_ball', 'output_when_not_observable', 'no_output_when_not_observable']:
            assert counts[key] == totals[key]
        visible = sum(p['phase_status'] == 'play' and gt[sid]['truth_visibility'] == 'visible' for sid, p in phases.items())
        assert visible == totals['play_visible']
        assert totals['pooled_visible_recall16px'] == counts['correct16px'] / visible
    effects = []
    additions = []
    unresolved = []
    for name in names:
        gains, losses = [], []
        for sid, p in phases.items():
            a, b = outcomes['current_temporal'][sid], outcomes[name][sid]
            gain = p['phase_status'] == 'play' and b == 'correct16px' and a != 'correct16px'
            loss = p['phase_status'] == 'play' and a == 'correct16px' and b != 'correct16px'
            assert int(pairs[name, sid]['gain']) == gain and int(pairs[name, sid]['loss']) == loss
            if gain:
                gains.append(sid)
                additions.append(dict(method=name, sample_id=sid, clip_id=p['clip_id'], local_frame=p['local_frame'], source_frame=p['source_frame']))
            if loss:
                losses.append(sid)
            base, candidate = predictions['current_temporal'][sid], predictions[name][sid]
            if int(base['visible']):
                assert all(base[k] == candidate[k] for k in ['visible', 'x_px', 'y_px', 'confidence'])
            if name == 'logistic10_oof' and p['phase_status'] == 'play' and b not in ['correct16px', 'no_output_when_not_observable']:
                unresolved.append(dict(sample_id=sid, clip_id=p['clip_id'], local_frame=p['local_frame'], source_frame=p['source_frame'], outcome=b, truth_visibility=gt[sid]['truth_visibility']))
        effects.append(dict(method=name, gained_correct=len(gains), lost_correct=len(losses), gain_clips=dict(Counter(phases[s]['clip_id'] for s in gains))))
    write(OUT / 'verified_play_gains.csv', additions)
    write(OUT / 'remaining_play_issues.csv', unresolved)
    snapshot = OUT / 'completed_snapshot'
    snapshot_hashes = {}
    for p in snapshot.iterdir():
        assert p.is_file() and sha(p) == sha(OUT / p.name)
        snapshot_hashes[p.name] = sha(p)
    assert before == {str(p): sha(p) for p in tracked}
    verification = dict(status='passed', human_phase_rows=123, journal_events=len(events),
        phase_counts=report['phase_counts'], frozen_outputs_reconciled=492, paired_effects=effects,
        original_outputs_preserved=True, prior_answers_unchanged=True,
        protected_input_hashes_verified=True, no_research_labels_written=True,
        completed_snapshot_hashes=snapshot_hashes, script_sha256=sha(Path(__file__)),
        remaining_candidate_missing_visible=sum(r['outcome'] == 'missing_visible_ball' for r in unresolved),
        remaining_candidate_output_when_not_observable=sum(r['outcome'] == 'output_when_not_observable' for r in unresolved),
        operational_adoption=False)
    dump(OUT / 'completed_verification.json', verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
