"""Record one scoped human sequence answer and describe unchanged position proxies."""
from pathlib import Path
from collections import defaultdict
import shutil
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b

PRIOR=b.HERE/'044_stroke_review'
OUT=b.HERE/'044_sequence_confirmed'

def main(redraw_only=False):
    retained={}
    if redraw_only:
        assert (OUT/'human_sequence_confirmation.json').exists()
        retained={n:b.sha(OUT/n) for n in ['human_sequence_confirmation.json','confirmed_stroke_sequence.csv','unchanged_reference_player_proxies.csv','reference_arrangement_diagnostics.csv','all_consecutive_reference_proxy_changes.csv']}
        shutil.copy2(OUT/'confirmed_sequence_arrangement.jpg',OUT/'confirmed_sequence_arrangement_layout_v1.jpg')
    else:
        assert not OUT.exists(),'Preserve existing confirmation snapshots.'
    oldhash=b.load(PRIOR/'publication_hashes.json');assert all(b.sha(PRIOR/n)==h for n,h in oldhash.items())
    records=b.read(PRIOR/'AI_stroke_records.csv');positions=b.read(PRIOR/'stroke_player_snapshot_proxies.csv')
    expected=['far_rear','near_rear','far_rear','near_rear','far_rear']
    assert [r['hitter_candidate'] for r in records]==expected
    OUT.mkdir(exist_ok=redraw_only)
    question='今回の044の動画で、サーブからの打球順は「F1（奥・紫枠）→N2（手前・青枠）→F1→N2→F1」の5打球で合っていますか？ 打球数と打球者だけの確認で、正確な接触フレームの指定は不要です。'
    receipt=dict(recorded_on_jst='2026-11-11',reviewer='human_user',rally='match1_rally_window_044',
        question_item_id='["request_user_input_async","call_91e1ef0825294eb186d21ad6ca302e8f",0]',
        question=question,answer='合っている',evidence_video=str(PRIOR/'044_AI_stroke_sequence_half_speed.mp4'),
        evidence_video_sha256=b.sha(PRIOR/'044_AI_stroke_sequence_half_speed.mp4'),
        confirmed_fields=['five_strokes_including_serve_in_displayed_sequence','hitter_order_F1_N2_F1_N2_F1'],
        not_confirmed_fields=['exact_contact_frame_or_AI_brackets','all_frame_tracking_identity','physical_foot_or_court_positions',
            'ball_center_accuracy','bounce_labels_or_bounce_times','landing_coordinates','shot_type','tactical_intention_or_cause_of_point'],
        prediction_visible_to_reviewer=True,independent_blind_accuracy_evaluation=False,
        review_actions=1,sequences=1,not_five_independent_rallies=True)
    b.dump(OUT/'human_sequence_confirmation.json',receipt)
    output=[]
    for r in records:
        output.append(dict(stroke_id=r['stroke_id'],ordinal=r['ordinal'],hitter_slot=r['hitter_candidate'],
            stroke_count_and_hitter_order_status='human_confirmed_displayed_sequence',confirmation_source='human_sequence_confirmation.json',
            source_frame_first=r['source_frame_first'],source_frame_last=r['source_frame_last'],reference_source_frame=r['reference_source_frame'],
            contact_interval_status=r['timing_status'],exact_contact_time_human_confirmed=0,
            reference_is_not_exact_contact_time=1,all_four_position_proxies_present=r['all_four_position_proxies_present'],
            hitter_court_proxy_x=r['hitter_court_proxy_x'],hitter_court_proxy_y=r['hitter_court_proxy_y'],
            physical_position_human_confirmed=0,all_frame_identity_human_confirmed=0))
    b.write(OUT/'confirmed_stroke_sequence.csv',output)
    # Keep the same explicit empty position row, no re-selection of reference times.
    b.write(OUT/'unchanged_reference_player_proxies.csv',positions)
    by=defaultdict(dict)
    for r in positions:by[r['stroke_id']][r['player_slot']]=r
    configs=[]
    for r in records:
        item=dict(stroke_id=r['stroke_id'],reference_source_frame=r['reference_source_frame'])
        for side in ['near','far']:
            slots=[s for s in b.SLOTS if s.startswith(side+'_')];pair=[by[r['stroke_id']][s] for s in slots]
            valid=all(v['observed']=='1' for v in pair);item[side+'_pair_observed']=int(valid)
            item[side+'_horizontal_gap_court_width_fraction']=abs(float(pair[0]['court_proxy_x'])-float(pair[1]['court_proxy_x'])) if valid else ''
            item[side+'_depth_gap_court_length_fraction']=abs(float(pair[0]['court_proxy_y'])-float(pair[1]['court_proxy_y'])) if valid else ''
        configs.append(item)
    b.write(OUT/'reference_arrangement_diagnostics.csv',configs)
    deltas=[]
    for prev,nxt in zip(records,records[1:]):
        for slot in b.SLOTS:
            a=by[prev['stroke_id']][slot];z=by[nxt['stroke_id']][slot];valid=a['observed']=='1' and z['observed']=='1'
            deltas.append(dict(from_stroke=prev['stroke_id'],to_stroke=nxt['stroke_id'],player_slot=slot,both_reference_points_observed=int(valid),
                delta_court_proxy_x=float(z['court_proxy_x'])-float(a['court_proxy_x']) if valid else '',
                delta_court_proxy_y=float(z['court_proxy_y'])-float(a['court_proxy_y']) if valid else '',
                meaning='difference_between_two_reference_bbox_proxies_not_continuous_path_or_physical_velocity'))
    b.write(OUT/'all_consecutive_reference_proxy_changes.csv',deltas)
    # A complete five-column comparison: no selecting only the largest displacement.
    img=Image.new('RGB',(1600,1080),(13,23,34));dr=ImageDraw.Draw(img)
    fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [17,19,21,24,29,34]}
    def text(x,y,s,size=21,color=(228,236,244)):dr.text((x,y),s,font=fonts[size],fill=color)
    text(24,14,'044：打球順の人手確認と、一打ごとの配置',34)
    text(25,68,'確認済み：サーブを含む5打球、F1 → N2 → F1 → N2 → F1',24,(105,235,175))
    text(25,111,'図の点は同じ代表時点の人物枠下端の投影。接触時刻・足元座標の正解ではありません。',21)
    for i,r in enumerate(records):
        px=24+i*314;left=px+57;top=238;cw=190;ch=425
        def cp(u,v):return left+cw*u,top+ch*(1-v)
        text(px+15,169,f'{i+1} '+('サーブ' if i==0 else '返球')+'：'+b.NAMES[r['hitter_candidate']],24,b.RGB[r['hitter_candidate']])
        text(px+15,674,f'代表 F{r["reference_source_frame"]}',17)
        dr.rectangle((left,top,left+cw,top+ch),outline=(125,160,180),width=2)
        for v in [.231,.5,.769]:dr.line([cp(0,v),cp(1,v)],fill=(210,225,235) if v==.5 else (70,110,135),width=2 if v==.5 else 1)
        for u in [.125,.875]:dr.line([cp(u,0),cp(u,1)],fill=(70,110,135))
        missing=[]
        for s in b.SLOTS:
            p=by[r['stroke_id']][s]
            if p['observed']!='1':missing.append(b.NAMES[s]);continue
            x,y=cp(float(p['court_proxy_x']),float(p['court_proxy_y']));c=b.RGB[s]
            dr.ellipse((x-7,y-7,x+7,y+7),fill=c);text(int(x+10),int(y-12),b.NAMES[s],19,c)
        text(px+12,705,'欠測：'+('・'.join(missing) if missing else 'なし'),19,(250,200,95) if missing else (210,225,235))
        near=configs[i]['near_horizontal_gap_court_width_fraction']
        text(px+12,744,f'手前2人の左右間隔：約{near*100:.0f}%',19)
    text(24,802,'配置から読めること（この1ラリーの記述）',29)
    text(28,853,'・N2の返球位置は、2打目から4打目の代表時点で右側へ移っています。',24)
    text(28,898,'・その2時点で、手前ペアの左右間隔も広がっています。奥F1は左側で返球しています。',24)
    text(28,958,'間隔の％はコート幅に対する代理点の差。実測距離・球の着地点・戦術的な意図ではありません。',19)
    text(28,1002,'打球順の確認を、全フレームの本人同定精度や座標精度の確認へ広げていません。',21,(250,205,110))
    img.save(OUT/'confirmed_sequence_arrangement.jpg',quality=95)
    # Numerical/provenance checks, separate from evidence of physical accuracy.
    assert len(output)==5 and [r['hitter_slot'] for r in output]==expected
    assert sum(r['observed']=='0' for r in positions)==1
    assert b.read(OUT/'unchanged_reference_player_proxies.csv')==positions
    for a,z in zip(records,output):
        for k in ['source_frame_first','source_frame_last','reference_source_frame','hitter_court_proxy_x','hitter_court_proxy_y']:assert a[k]==z[k]
        assert z['exact_contact_time_human_confirmed']==0 and z['physical_position_human_confirmed']==0
    assert all(b.sha(PRIOR/n)==h for n,h in oldhash.items())
    b.dump(OUT/'verification.json',dict(old_publication_files_unchanged=len(oldhash),
        human_sequence_review_actions=1,human_reviewed_rallies=1,confirmed_stroke_actor_rows=5,
        newly_confirmed_exact_contact_times=0,newly_confirmed_position_points=0,
        all_reference_times_coordinates_and_missing_unchanged=True,missing_reference_position_rows=1,
        full_tracking_accuracy_not_inferred=True,new_independent_accuracy_evaluation=False))
    b.dump(OUT/'source_hashes.json',{str(n):b.sha(n) for n in [Path(__file__),PRIOR/'publication_hashes.json',
        PRIOR/'AI_stroke_records.csv',PRIOR/'stroke_player_snapshot_proxies.csv',PRIOR/'AI_event_proposals.json']})
    shutil.copy2(Path(__file__),OUT/'executed_source.py')
    assert all(b.sha(OUT/n)==h for n,h in retained.items()),'Presentation redraw must preserve the confirmation and numerical records byte-for-byte.'
    print('Recorded one human sequence confirmation;5actors confirmed,0exact contact/position labels.',flush=True)

if __name__=='__main__':
    import sys
    main(redraw_only='--redraw-only' in sys.argv)
