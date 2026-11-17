"""Record displayed stroke-order confirmation without validating trajectories."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT / '056_reviewed_exclusion_F154780'
OUT = ROOT / '056_sequence_confirmed'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=False)
    protected = json.loads((PRIOR / 'publication_hashes.json').read_text(encoding='utf-8-sig'))
    assert all(sha(PRIOR / name) == digest for name, digest in protected.items())
    strokes = read(PRIOR / 'AI_stroke_records.csv')
    names = {'near_rear': 'N2', 'near_front': 'N1', 'far_rear': 'F1', 'far_front': 'F2'}
    sequence = [names[r['hitter_candidate']] for r in strokes]
    assert sequence == ['N2','F1','N1','F1','N1','F2','N2','F1','N2','F2','N1','F1','N1']
    dump('human_sequence_confirmation.json', {
        'recorded_on': '2026-11-11', 'rally': '056',
        'user_message': '13打球の順番は合ってる．ボレーやスマッシュを打った後の球の軌道を追うのは難しい？',
        'confirmation_quote': '13打球の順番は合ってる．',
        'confirmed_scope': ['displayed stroke count: 13', 'displayed hitter order'],
        'hitter_order': sequence,
        'not_confirmed': ['exact impact frames or intervals', 'shot types including volley/smash',
                          'ball centers, bounces or trajectories', 'continuous player identity accuracy'],
        'review_method': 'human confirmation after viewing model-assisted presentation; not blinded independent trajectory ground truth',
        'reviewed_rallies': 1, 'review_actions': 1,
        'context_video': str(PRIOR / '056_transfer_bounces_and_routes.mp4'),
        'context_video_sha256': sha(PRIOR / '056_transfer_bounces_and_routes.mp4'),
        'prior_wrong_candidate_exclusion_preserved': 154780,
    })
    confirmed = []
    for r in strokes:
        row = dict(r)
        row['prior_status'] = row['status']
        row['status'] = 'human_confirmed_displayed_count_and_hitter_order_only'
        row['hitter_display_name'] = names[r['hitter_candidate']]
        confirmed.append(row)
    with (OUT / 'stroke_order_confirmed.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(confirmed[0]))
        writer.writeheader()
        writer.writerows(confirmed)
    # Descriptive output availability after each provisional impact interval.
    # These windows and counts are not correctness/visibility annotations.
    path = read(PRIOR / 'observed_and_estimated_image_path.csv')
    windows = []
    for i, r in enumerate(strokes):
        first = int(r['source_frame_last']) + 2
        last = min(first + 16, int(strokes[i+1]['source_frame_first']) - 2 if i+1 < len(strokes) else 154880)
        rows = [p for p in path if first <= int(p['source_frame']) <= last]
        windows.append({'stroke_id': r['stroke_id'], 'hitter': names[r['hitter_candidate']],
                        'first': first, 'last': last,
                        'output_kinds': dict(Counter(p['kind'] for p in rows)),
                        'outside_existing_annotation_range': sum(int(p['source_frame']) > 154838 for p in rows)})
    dump('post_impact_output_inventory.json', {
        'purpose': 'descriptive output availability only; not accuracy or physical ball visibility',
        'impact_intervals_still_AI_provisional': True,
        'sampling': 'source frame stride 2, nominal source FPS 60000/1001',
        'windows': windows,
    })
    assert all(sha(PRIOR / name) == digest for name, digest in protected.items())
    dump('verification.json', {'prior_publication_files_unchanged': len(protected),
                             'confirmed_rows': len(confirmed), 'source_sequence_exact': True,
                             'timing_and_coordinate_fields_unchanged': all(
                                 all(c[k] == r[k] for k in r if k != 'status')
                                 for c, r in zip(confirmed, strokes)),
                             'ball_or_route_files_changed': 0})
    (OUT / 'README.md').write_text(
        '# 056：13打球の順序確認\n\n'
        '本人の「13打球の順番は合ってる．」を、打球数と表示された打球者順の確認として記録。\n\n'
        + ' → '.join(sequence) + '\n\n'
        '接触フレーム、球位置、球種、バウンド点、軌道の正確さは、この回答で確認されたものではない。'
        '既存公開物とF154780の別物体除外は変更していない。\n\n'
        'post_impact_output_inventory.json は暫定接触区間直後の出力の有無のみを数えた資料。'
        '本人注釈の範囲外と未取得を区別し、検出精度として扱わない。\n', encoding='utf-8')
    dump('publication_hashes.json', {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print(json.dumps({'output': str(OUT), 'sequence': sequence, 'post_impact_inventory': windows}, ensure_ascii=False))

if __name__ == '__main__':
    main()
