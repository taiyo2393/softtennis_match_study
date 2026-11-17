"""Apply two positive and one negative candidate review; preserve source labels."""
from pathlib import Path
import shutil
import build_044_spatiotemporal_pilot as b

OUT=b.HERE/'056_crop_reviewed'
TRIAL=b.HERE/'056_crop_reacquisition_trial'
END=b.HERE/'056_temporal_reacquisition_confirmed'
BASE=b.PRIOR/'rally_repair_056/completed_review'
ITEM='["request_user_input_async","call_f75d0f1cd35a4d3093a44524f0039f81",0]'

def main():
    OUT.mkdir(exist_ok=False)
    protected={}
    for folder in [TRIAL,END,b.HERE/'056_reviewed_exclusion_F154780']:
        for name,digest in b.load(folder/'publication_hashes.json').items():
            assert b.sha(folder/name)==digest;protected[str(folder/name)]=digest
    original_path=BASE/'assisted_states_sampled.csv';protected[str(original_path)]=b.sha(original_path)
    raw=b.read(TRIAL/'candidate_rows.csv');by={int(r['source_frame']):r for r in raw}
    answers={154270:('A','actual_ball'),154324:('B','other_object'),154780:('C','actual_ball')}
    reviews=[]
    for f,(letter,decision) in answers.items():
        r=by[f];assert r['kind']=='crop_heatmap_candidate_unconfirmed'
        reviews.append(dict(letter=letter,source_frame=f,candidate_x_px=r['candidate_x_px'],
            candidate_y_px=r['candidate_y_px'],decision=decision,question_item_id=ITEM,
            exact_center_validated=0,ball_absence_label=0))
    b.write(OUT/'scoped_candidate_review.csv',reviews)
    b.dump(OUT/'human_candidate_review.json',dict(
        recorded_on='2026-11-11',rally='056',question_item_id=ITEM,
        question='今回の056の確認動画で、青◇のA（F154270）・B（F154324）・C（F154780）は実球を囲めていますか？ 違うものがあればA・B・Cで教えてください。Cの赤×は以前に別物体と確認された位置で、今回確認するのは別位置の青◇です。',
        answer='Bは違うACは合ってる',candidate_reviews=reviews,
        video_path=str(TRIAL/'056_crop_three_candidates.mp4'),video_sha256=b.sha(TRIAL/'056_crop_three_candidates.mp4'),
        scope='A and C enclose the actual rally ball. B is a wrong candidate. No new exact centers, ball absence, or continuous trajectory truth.',
        F154780='Earlier candidate1102,579 remains rejected; the new936.875,529.875 candidate is separately confirmed.',
        workflow='One prediction-visible review of3selected candidates in one repeatedly inspected rally. Not independent precision estimation.',
    ))
    candidate_rows=[]
    for r in raw:
        z=dict(r);f=int(r['source_frame']);z['prior_review_status']=r['review_status']
        if f in answers:z['review_status']='human_confirmed_actual_ball' if answers[f][1]=='actual_ball' else 'human_rejected_other_object'
        z['new_review_source']=ITEM if f in answers else ''
        candidate_rows.append(z)
    b.write(OUT/'crop_candidate_rows_with_review.csv',candidate_rows)
    original=b.read(original_path);observations=[]
    for r in original:
        f=int(r['source_frame']);was=r['observation_present']=='1'
        z=dict(r)
        z.update(current_observation_present=r['observation_present'],
                 current_x_px=r['x_px'] if was else '',current_y_px=r['y_px'] if was else '',
                 current_coordinate_source=r['coordinate_source'] if was else 'unlocated_not_absence',
                 current_review_source='historical_assisted_review',current_exact_center_validated=0)
        if f in [154270,154780]:
            z.update(current_observation_present='1',current_x_px=by[f]['candidate_x_px'],current_y_px=by[f]['candidate_y_px'],
                     current_coordinate_source='crop_heatmap_candidate_human_confirmed_ball_enclosure',current_review_source=ITEM)
        # B does not imply that the physical ball was absent; its old missing label remains.
        if f==154324:assert z['current_observation_present']=='0'
        observations.append(z)
    ending=b.read(END/'post_H13_ball_path_reviewed.csv')
    for r in ending:
        present=r['observation_present']=='1'
        observations.append(dict(source_frame=r['source_frame'],clock='legacy_edited_match1',
            current_observation_present=r['observation_present'],current_x_px=r['x_px'] if present else '',
            current_y_px=r['y_px'] if present else '',current_coordinate_source=r['coordinate_source'] if present else 'unlocated_not_absence',
            current_review_source=r['confirmation_source'],current_exact_center_validated=0,
            original_annotation_range='outside_original376times'))
    observations.sort(key=lambda r:int(r['source_frame']))
    b.write(OUT/'056_reviewed_ball_observations.csv',observations)
    current={int(r['source_frame']):r for r in observations}
    assert len(current)==len(observations)==404
    assert all(all(current[int(r['source_frame'])][k]==v for k,v in r.items()) for r in original)
    changed=[int(r['source_frame']) for r in original if
        (current[int(r['source_frame'])]['current_observation_present'],
         current[int(r['source_frame'])]['current_x_px'],current[int(r['source_frame'])]['current_y_px'])!=
        (r['observation_present'],r['x_px'] if r['observation_present']=='1' else '',r['y_px'] if r['observation_present']=='1' else '')]
    assert changed==[154270,154780]
    assert sum(r['current_observation_present']=='1' for r in observations)==354
    for r in ending:
        z=current[int(r['source_frame'])]
        assert z['current_observation_present']==r['observation_present']
        assert z['current_x_px']==r['x_px'] and z['current_y_px']==r['y_px']
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    b.dump(OUT/'verification.json',dict(
        protected_input_hashes=protected,old_publications_and_original_fields_unchanged=True,
        accepted_new_candidates=[154270,154780],rejected_new_candidates=[154324],
        changed_current_positions_within_original_annotations=changed,
        original_other333observations_exact=True,ending28rows_unchanged=True,
        combined_annotation_times=404,current_ball_enclosing_positions=354,current_unlocated_times=50,
        new_absence_labels=0,new_interpolated_positions=0,new_exact_center_labels=0,
        note='Mixed sampling: original376times at stride2 plus28end times at stride1. Counts are not continuous coverage or accuracy.',
    ))
    b.dump(OUT/'decision.json',dict(
        status='review_complete_two_candidates_accepted_one_rejected',
        automatic_tracking_replacement=False,automatic_weak_crop_fill=False,
        reason='B was a wrong object although exactly1component passed the fixed temporal search gate. Cropping and temporal proximity do not establish ball identity.',
        operational_scope='Use the two reviewed points as explicit local repairs. Leave unreviewed crop outputs unadopted; retain old successful positions.',
        threshold_tuning_from_this_reply=False,other_match_effectiveness_established=False,
        candidate_enclosure_is_not_exact_center_truth=True))
    (OUT/'README.md').write_text(
        '# 056：局所画像候補の本人確認\n\n'
        '回答「Bは違うACは合ってる」を保存した。A（F154270）とC（F154780）を実球を囲む位置として追加し、'
        'B（F154324）は別物体として棄却する。Bの真の球位置は未取得のまま。\n\n'
        'F154780では以前の1102,579の候補は引き続き誤検出として保持する。'
        '今回確認されたのは別位置936.875,529.875の候補であり、過去の回答を書き換えない。\n\n'
        '056_reviewed_ball_observations.csv は、原376時点の全列を残したまま現在採用する位置を別列に追加し、'
        '終盤28時点を併合した404時点の観測表。位置あり354、未取得50。'
        '元の範囲は2フレーム間隔、終盤は1フレーム間隔であり、連続追跡率や精度ではない。'
        '既存の補間推定点はこの観測表に混ぜていない。\n\n'
        '実球同定が確認された2点を局所修正として採用する。局所画像方式は、前後位置の近くでも別物体を拾ったため、'
        '無確認で欠測を埋める方法としては採用しない。'
        '1ラリーの選択された3候補の確認結果から、一般的な検出精度を推定しない。\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'executed_confirmation.py')
    b.dump(OUT/'publication_hashes.json',{p.name:b.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print('Accepted A,C; rejected B; original columns and publications preserved; combined404times:354positions,50unlocated.')

if __name__=='__main__':main()
