"""Add four user-confirmed ball-enclosing candidates; retain nine missing frames."""
from pathlib import Path
import shutil
import build_044_spatiotemporal_pilot as b

OUT=b.HERE/'056_temporal_reacquisition_confirmed'
TRIAL=b.HERE/'056_temporal_reacquisition_trial'
PREVIOUS=b.HERE/'056_post_hit_confirmation'
ITEM='["request_user_input_async","call_55bf5f0e26d84057b8d7616ba79df616",0]'

def main():
    OUT.mkdir(exist_ok=False)
    protected={}
    for folder in [TRIAL,PREVIOUS]:
        for name,digest in b.load(folder/'publication_hashes.json').items():
            assert b.sha(folder/name)==digest
            protected[str(folder/name)]=digest
    raw=b.read(TRIAL/'proposed_path.csv')
    previous=b.load(PREVIOUS/'human_video_review.json')
    known=set(previous['derived_confirmed_candidate_frames'])
    expected={154861,154862,154873,154878}
    actual={int(r['source_frame']) for r in raw if r['kind']=='weak_heatmap_candidate_unconfirmed'}
    assert actual==expected
    video=TRIAL/'056_four_reacquisition_candidates.mp4'
    b.dump(OUT/'human_candidate_confirmation.json',dict(
        recorded_on='2026-11-11',rally='056',question_item_id=ITEM,
        question='056の新しい確認動画で、青◇の4候補（F154861・154862・154873・154878）は実球を囲めていますか？ 一部だけ違う場合はフレーム番号を教えてください。緑○は前回確認済みなので再確認は不要です。',
        answer='4候補とも実球',confirmed_source_frames=sorted(expected),
        video_path=str(video),video_sha256=b.sha(video),
        scope='The four displayed blue candidates enclose the actual rally ball.',
        not_confirmed=['exact ball centers or pixel error','ball position or physical absence in missing frames',
                       'exact contact/bounce timing or position','3D ball flight','independent accuracy on other rallies'],
        review_actions=1,reviewed_rallies=1,
        workflow='Prediction-visible review of four selected candidates from a previously reviewed development rally. Not blinded independent framewise ground truth.',
    ))
    reviewed=[];observations=[]
    for r in raw:
        f=int(r['source_frame']);new=f in expected;old=f in known
        assert not(new and old)
        present=new or old
        z=dict(r)
        z['prior_review_status']=r['review_status']
        z['review_status']='human_confirmed_ball_enclosed' if present else 'unlocated_not_absence'
        z['confirmation_source']=ITEM if new else previous['question_item_id'] if old else ''
        z['exact_center_validated']=0
        reviewed.append(z)
        observations.append(dict(source_frame=f,observation_present=int(present),
            x_px=r['proposed_x_px'] if present else '',y_px=r['proposed_y_px'] if present else '',
            coordinate_source='original_model_candidate_user_window_confirmed' if old else
                              'weak_heatmap_temporal_gate_candidate_user_confirmed' if new else '',
            ball_identity_review='candidate_encloses_ball' if present else 'unlocated_not_absence',
            confirmation_source=z['confirmation_source'],exact_center_validated=0,
            continuous_trajectory_validated=0,court_x='',court_y=''))
    b.write(OUT/'candidate_rows_with_confirmation.csv',reviewed)
    b.write(OUT/'post_H13_ball_path_reviewed.csv',observations)
    assert sum(r['observation_present'] for r in observations)==19
    missing=[r['source_frame'] for r in observations if not r['observation_present']]
    assert missing==list(range(154863,154869))+list(range(154874,154877))
    assert all(all(z[k]==r[k] for k in r if k!='review_status') for z,r in zip(reviewed,raw))
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    b.dump(OUT/'verification.json',dict(
        protected_input_hashes=protected,old_publications_unchanged=True,
        scope_source_frames=[154853,154880],source_frames=28,
        original_user_confirmed_candidates_unchanged=15,new_user_confirmed_candidates=4,
        ball_enclosing_candidates_after_review=19,remaining_unlocated_frames=missing,
        coordinates_changed=0,new_exact_center_labels=0,new_interpolated_positions=0,
        new_absence_labels=0,independent_accuracy_established=False,
    ))
    (OUT/'README.md').write_text(
        '# 056：再取得4候補の本人確認を反映\n\n'
        '回答「4候補とも実球」を保存。F154861・154862・154873・154878の候補を、'
        '実球を囲むことが確認された点として追加した。正確な中心座標の教師ラベルとは扱わない。\n\n'
        'post_H13_ball_path_reviewed.csv はF154853〜154880の28時点。'
        '以前の確認済み15点を保持し、今回の4点を追加して19点となった。'
        'F154863〜154868、F154874〜154876の9時点は未取得のまま。'
        '未取得は球の不在を意味しない。補間点・実測扱いの推定点は追加していない。\n\n'
        '今回の結果は、同じラリーの既知の欠測に対する探索と表示付き確認によるもの。'
        '4候補の有用性は確認できたが、別ラリーでの誤検出率・一般的な追跡精度改善は未検証。'
        '同じ設定の他区間への適用では別物体への乗り換えも評価する必要がある。\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'executed_confirmation.py')
    b.dump(OUT/'publication_hashes.json',{p.name:b.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print('Confirmed 4 new candidates; retained 15 prior candidates; 9 frames remain unlocated. Prior publications unchanged.')

if __name__=='__main__':main()
