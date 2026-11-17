"""Scope the user's video review to displayed candidates after the last impact interval."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT/'056_post_hit_trial'
OUT=ROOT/'056_post_hit_confirmation'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    OUT.mkdir(exist_ok=False)
    protected=load(PRIOR/'publication_hashes.json')
    assert all(sha(PRIOR/name)==digest for name,digest in protected.items())
    last=read(ROOT/'056_sequence_confirmed/stroke_order_confirmed.csv')[-1]
    assert last['stroke_id']=='H13'
    end=int(last['source_frame_last'])
    original=read(PRIOR/'unreviewed_tail_candidates.csv');derived=[]
    for r in original:
        z=dict(r);f=int(r['source_frame'])
        positive=f>end and r['usable_candidate']=='1'
        z['window_review_status']=('human_confirmed_ball_enclosed_in_displayed_candidate_window'
            if positive else 'not_individually_resolved_by_this_reply')
        z['exact_center_validated']='0'
        z['no_output_is_ball_absence_label']='0'
        derived.append(z)
    confirmed=[int(r['source_frame']) for r in derived if r['window_review_status'].startswith('human_confirmed')]
    dump('human_video_review.json',dict(
        recorded_on='2026-11-11',rally='056',
        question_item_id='["request_user_input_async","call_9bf3f6f0e9d04107be409ec2ad31dd59",0]',
        question='今回の056終盤動画（056_last_two_hits_model_candidates.mp4）で、最後の打球後に出る橙丸は実球を囲めていますか？ 別物体へ移っている場合は、おおよそのフレームか場面を教えてください。',
        answer='実球を囲めている',
        reviewed_video=str(PRIOR/'056_last_two_hits_model_candidates.mp4'),
        reviewed_video_sha256=sha(PRIOR/'056_last_two_hits_model_candidates.mp4'),
        confirmed_scope='Displayed orange candidates after the final stroke enclose the actual rally ball; one continuous video-window review.',
        conservative_row_application='Apply only to output-present frames strictly after the end of the existing AI H13 contact interval. Do not infer precise contact time from this answer.',
        AI_contact_interval_end=end,derived_confirmed_candidate_frames=confirmed,
        review_actions=1,reviewed_rallies=1,
        not_confirmed=['exact ball-center coordinates or error', 'ball locations or absence during missing outputs',
                       'exact impact or bounce timing and location', 'candidate correctness before the last stroke',
                       '3D trajectory', 'independent detector accuracy across rallies'],
        workflow='Prediction-visible video review; not blinded framewise independent center annotation.',
    ))
    with (OUT/'tail_candidates_with_scoped_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(derived[0]));writer.writeheader();writer.writerows(derived)
    assert all(all(z[k]==r[k] for k in r) for z,r in zip(derived,original))
    assert all(sha(PRIOR/name)==digest for name,digest in protected.items())
    dump('verification.json',dict(prior_publication_files_unchanged=len(protected),
        original_tail_rows_preserved=len(original),all_original_columns_exact=True,
        post_interval_displayed_candidates_receiving_window_review=len(confirmed),
        human_review_actions=1,new_exact_center_labels=0,new_absence_labels=0,
        new_interpolated_or_predicted_positions=0))
    (OUT/'README.md').write_text(
        '# 056：最後の打球後の候補を本人確認\n\n'
        '回答「実球を囲めている」を、表示された橙丸の実球同定に関する区間確認として保存した。\n\n'
        f'行への反映は保守的に、H13のAI暫定接触区間末尾F{end}より後の出力あり候補'
        f'{len(confirmed)}時点に限定。接触付近とそれ以前の候補は、この回答による個別確認として扱わない。'
        '1ラリー・1回の動画確認であり、各時点を独立試行として精度集計しない。\n\n'
        '位置座標と元出力は全て保持。欠測の球位置、球の不在、正確な中心、'
        '接触・バウンドの時刻や位置、3D軌道は確定していない。\n',encoding='utf-8')
    dump('publication_hashes.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print(json.dumps(load(OUT/'verification.json'),ensure_ascii=False))

if __name__=='__main__':main()
