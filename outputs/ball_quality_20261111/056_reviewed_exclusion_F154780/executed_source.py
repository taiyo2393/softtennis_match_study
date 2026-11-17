"""One scoped human candidate rejection as an overlay, never a ball-absence label."""
from pathlib import Path
from collections import Counter
import math,shutil,sys
import numpy as np,cv2
from PIL import ImageDraw
import build_044_spatiotemporal_pilot as b
import render_056_transfer_routes as renderer

PRIOR=b.HERE/'056_route_transfer'
OUT=b.HERE/'056_reviewed_exclusion_F154780'
TARGET=154780

def main():
    assert not OUT.exists(),'Retain prior correction snapshots.'
    prior_hashes=b.load(PRIOR/'publication_hashes.json');assert all(b.sha(PRIOR/n)==h for n,h in prior_hashes.items())
    OUT.mkdir()
    question='056の短い拡大動画「F154780_ball_identity_review.mp4」で、停止時の赤丸はラリーの実球を囲んでいますか？ この1点だけの確認です。'
    b.dump(OUT/'human_candidate_rejection.json',dict(recorded_on_jst='2026-11-11',reviewer='human_user',
        question_item_id='["request_user_input_async","call_137e7f3d15f047d8b96f844bc0ec36b8",0]',question=question,answer='別物体を囲んでいる',
        source_frame=TARGET,candidate_image_x_px=1102.,candidate_image_y_px=579.,
        evidence_video=str(PRIOR/'F154780_ball_identity_review.mp4'),evidence_video_sha256=b.sha(PRIOR/'F154780_ball_identity_review.mp4'),
        scope='Reject the one displayed candidate as another object. No true ball center, precise impact time, neighboring frame correctness or13stroke order confirmed.',
        ball_absent_label_created=False,zero_target_training_mask_created=False,true_center_created=False,
        automatic_filter_accuracy_established=False,review_actions=1))
    source=b.read(PRIOR/'observed_and_estimated_image_path.csv');actual=[]
    for r in source:
        q=dict(r)
        if int(r['source_frame'])==TARGET:
            assert r['kind']=='observed_warning' and r['original_image_x_px']=='1102.0' and r['original_image_y_px']=='579.0'
            assert r['estimated_image_x_px']==r['estimated_image_y_px']==r['estimate_source']==''
            q.update(display_image_x_px='',display_image_y_px='',kind='missing')
        actual.append(q)
    b.write(OUT/'observed_and_estimated_image_path.csv',actual)
    unchanged=['player_court_proxies.csv','AI_stroke_records.csv','bounce_court_points.csv','schematic_court_routes.csv']
    for name in unchanged:shutil.copy2(PRIOR/name,OUT/name)
    b.write(OUT/'scoped_exclusion.csv',[dict(source_frame=TARGET,old_candidate_x=1102.,old_candidate_y=579.,
        candidate_identity='other_object_human_confirmed',current_usable_ball_position=0,replacement_x='',replacement_y='',
        observation_status='unlocated_ball_not_physical_absence',confirmation_source='human_candidate_rejection.json')])
    # Same direction-change rule, excluding this rejected candidate from valid ball observations.
    balls=b.read(PRIOR/'ball_image_observations.csv')
    points={int(r['source_frame']):np.array([float(r['image_x_px']),float(r['image_y_px'])]) for r in balls if r['observation_present']=='1' and int(r['source_frame'])!=TARGET}
    candidates=[]
    for f in sorted(points):
        if not all(f+d in points for d in [-4,-2,0,2,4]):continue
        vin=(points[f]-points[f-4])/4;vout=(points[f+4]-points[f])/4;norms=[np.linalg.norm(v) for v in [vin,vout]]
        if min(norms)<.5:continue
        angle=math.degrees(math.acos(float(np.clip(np.dot(vin,vout)/np.prod(norms),-1,1))))
        if angle>=45:candidates.append(dict(source_frame=f,turn_degrees=angle,ball_x=float(points[f][0]),ball_y=float(points[f][1]),event_status='direction_change_only_not_hit_or_bounce_GT'))
    kept=[]
    for r in sorted(candidates,key=lambda r:(-r['turn_degrees'],r['source_frame'])):
        if all(abs(r['source_frame']-z['source_frame'])>=12 for z in kept):kept.append(r)
    kept.sort(key=lambda r:r['source_frame'])
    for i,r in enumerate(kept,1):r['candidate_id']=f'REV_C{i:02}'
    b.write(OUT/'direction_change_candidates_after_exclusion.csv',kept)
    models=b.load(PRIOR/'bounce_fit_models.json');assert all(TARGET not in m['fit_source_frames'] for m in models)
    b.dump(OUT/'protocol.json',dict(date_jst='2026-11-11',purpose='Apply exactly one human-confirmed other-object rejection as a separate display/analysis overlay.',
        rule='Only source154780 display point becomes unlocated. Keep original raw observation fields and all other rows. Do not replace it with adjacent interpolation. Redraw same401source frames with confirmation status.',
        source_hashes={str(p):b.sha(p) for p in [Path(__file__),b.HERE/'render_056_transfer_routes.py',PRIOR/'publication_hashes.json',PRIOR/'observed_and_estimated_image_path.csv']},
        protected_prior_files_unchanged=True,other_stroke_bounce_and_player_data_unchanged=unchanged,
        original_assisted_answers_retained=True,new_human_true_ball_centers=0))
    # Reuse numerical/layout renderer; only wording for this confirmed rejection changes.
    text_before=ImageDraw.ImageDraw.text
    replacements={
        '056：同じ方法で選手配置・配球・球軌跡を表示':'056：別物体1点を除外した修正版',
        '警告：F154780の球位置に単独の飛び。元データを残し、線は結ばない。':'F154780：別物体と確認済み。球位置は未取得のまま、代わりの点は埋めない。',
        '緑＝整備済み観測 ／ 黄破線＝補完推定 ／ 赤丸＝位置の警告':'緑＝整備済み観測 ／ 黄破線＝補完推定 ／ 別物体と確認した1点は未取得'}
    def reviewed_text(self,xy,text,*args,**kwargs):return text_before(self,xy,replacements.get(text,text),*args,**kwargs)
    renderer.OUT=OUT;ImageDraw.ImageDraw.text=reviewed_text
    try:renderer.render()
    finally:ImageDraw.ImageDraw.text=text_before
    # Exact comparisons protect against spreading one answer into nearby frames/labels.
    written=b.read(OUT/'observed_and_estimated_image_path.csv');changes=[]
    for a,z in zip(source,written):
        if a!=z:changes.append(int(a['source_frame']))
        if int(a['source_frame'])!=TARGET:assert a==z
        for k in ['original_observation_present','original_image_x_px','original_image_y_px','estimated_image_x_px','estimated_image_y_px','estimate_source']:
            assert a[k]==z[k]
    assert changes==[TARGET]
    for name in unchanged:assert b.sha(PRIOR/name)==b.sha(OUT/name)
    oldturns=b.read(PRIOR/'direction_change_candidates.csv');newtimes={r['source_frame'] for r in kept};oldtimes={int(r['source_frame']) for r in oldturns}
    b.dump(OUT/'direction_candidate_changes.json',dict(removed=sorted(oldtimes-newtimes),added=sorted(newtimes-oldtimes),
        numerical_rule_changed=False,meaning='Derived candidate availability only, not event precision/recall or new event GT.'))
    counts=dict(Counter(r['kind'] for r in written));assert counts=={'missing':47,'observed_assisted':333,'estimated':21}
    cap=cv2.VideoCapture(str(OUT/'056_transfer_bounces_and_routes.mp4'));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(1080,1920,3);n+=1
    cap.release();assert n==401
    assert b.read(OUT/'video_source_frame_map.csv')==b.read(PRIOR/'video_source_frame_map.csv')
    assert all(b.sha(PRIOR/n)==h for n,h in prior_hashes.items())
    b.dump(OUT/'verification.json',dict(prior_publication_files_unchanged=len(prior_hashes),changed_display_frames=changes,
        all_other400rows_exact=True,original_observation_fields_retained=True,estimated21positions_unchanged=True,
        all_player_stroke_bounce_route_tables_exact=True,display_counts=counts,
        reviewed_candidate_rejections=1,new_true_ball_centers=0,ball_absence_labels_created=0,
        replacement_positions_created=0,video_source_map_exact=True,video_frames_decoded=n,
        frame154780_not_used_in_existing_bounce_fits=True,independent_automatic_accuracy_not_established=True))
    shutil.copy2(Path(__file__),OUT/'executed_source.py')
    print('Applied1scoped rejection; all other400rows/21estimates/player/event tables unchanged;401video frames.',flush=True)

if __name__=='__main__':main()
