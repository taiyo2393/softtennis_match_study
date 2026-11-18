"""Review2c preservation, comparisons and exact composite video checks."""
from __future__ import annotations

import json
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

import build_court_view as b

OUT=b.OUT
BACKUP=OUT/'review2c_before'


def snapshot():
    manifest=OUT/'review2c_before.json'
    if manifest.exists():return json.loads(manifest.read_text(encoding='utf-8'))
    BACKUP.mkdir(exist_ok=True)
    names=['cleaning_summary.json','review2_report.md','review2_far_depth_counts.csv',
           'review2_reentry_intervals.csv','hitter_auto.csv','identity_summary.json','wrist_identity.csv','README.md',
           'review2b_envelope_stats.csv','review2b_envelope_intervals.csv']
    names += [f'{prefix}_p{p:02d}.csv' for p in range(7,12)
              for prefix in ('cleaned_player_positions','identity_positions','identity_map','video_source_frame_map')]
    for name in names:shutil.copyfile(OUT/name,BACKUP/name)
    value=dict(version='review2c_20261119',sha256={n:b.sha(BACKUP/n) for n in names},
               git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip())
    b.dump(manifest,value)
    return value


def runs(frames):
    result=[]
    for f in sorted(frames):
        f=int(f)
        if not result or f!=result[-1][-1]+1:result.append([])
        result[-1].append(f)
    return result


def selected(table,summary,p):
    mask=pd.Series(False,index=table.index)
    for seg in summary['points'][str(p)]['rendering']['segments']:
        mask |= table.source_frame.between(seg['start_frame'],seg['end_frame'])
    return mask


def finalize(s,write=True):
    import review2 as r2
    prior=json.loads((BACKUP/'cleaning_summary.json').read_text(encoding='utf-8'))
    for key,value in prior.items():
        if key.startswith('review2b'):s[key]=value
    counts=pd.read_csv(BACKUP/'review2_far_depth_counts.csv')
    resets=[];intervals=[];identity_changes=[]
    for p in range(7,12):
        table=pd.read_csv(OUT/f'cleaned_player_positions_p{p:02d}.csv')
        old=pd.read_csv(BACKUP/f'cleaned_player_positions_p{p:02d}.csv')
        for slot in b.SLOTS[2:]:
            now=table[selected(table,s,p)&table.player_slot.eq(slot)].set_index('source_frame')
            prev=old.set_index(['source_frame','player_slot']).xs(slot,level='player_slot').loc[now.index]
            ref=pd.read_csv(r2.env.BACKUP/f'cleaned_player_positions_p{p:02d}.csv')
            ref=ref.set_index(['source_frame','player_slot']).xs(slot,level='player_slot').loc[now.index]
            valid=now.status.isin(b.VALID)&now.court_y.notna()
            ov=prev.status.isin(b.VALID)&prev.court_y.notna()
            rv=ref.status.isin(b.VALID)&ref.court_y.notna()
            at=counts.point.eq(p)&counts.player_slot.eq(slot)
            vals=dict(review2c_above_25_3m=int((valid&now.court_y.mul(23.77).gt(25.3)).sum()),
                      review2c_offscreen_held_above_25_3m=int((valid&now.status.eq(r2.HOLD)&now.court_y.mul(23.77).gt(25.3)).sum()),
                      review2c_valid_frames=int(valid.sum()),review2c_new_valid_frames=int((~ov&valid).sum()),
                      review2c_lost_valid_frames=int((ov&~valid).sum()),
                      review2c_lost_valid_frames_vs_review2=int((rv&~valid).sum()))
            for key,value in vals.items():counts.loc[at,key]=value
        for q in table[table.speed_reset_frame.notna()].itertuples():
            resets.append(dict(point=p,player=b.NAMES[q.player_slot],player_slot=q.player_slot,
                               rally_no=int(q.rally_no),start_frame=int(q.speed_reject_start_frame),
                               rejected_end_frame=int(q.source_frame)-1,reset_frame=int(q.speed_reset_frame),
                               rejection_span_source_frames=int(q.source_frame-q.speed_reject_start_frame),
                               accepted_depth_m=float(q.court_y*23.77),base_status=q.speed_reset_base_status))
        prev=pd.read_csv(BACKUP/f'identity_map_p{p:02d}.csv').fillna('')
        now=pd.read_csv(OUT/f'identity_map_p{p:02d}.csv').fillna('')
        merged=prev.merge(now,on=['source_frame','track_id'],suffixes=('_review2b','_review2c'),how='outer')
        for q in merged[merged.player_slot_review2b.ne(merged.player_slot_review2c)].to_dict('records'):
            identity_changes.append(dict(point=p,**q))
        if p==7:
            g=table[selected(table,s,p)&table.player_slot.eq('far_rear')&table.status.isin(b.VALID)&table.court_y.mul(23.77).gt(25.3)]
            for run in runs(g.source_frame):
                z=g[g.source_frame.isin(run)]
                occ=int(z.lower_occluded.eq(True).sum());held=int(z.status.eq(r2.HOLD).sum())
                cause=('画面外の最終位置保持が超過を継続（保持開始時の足元推定に偏りの可能性）' if held>len(z)/2 else
                       '下隠れの深さアンカー補間が主体。F23672はF23643/23742を参照。アンカー枠の足元偏り・実後退の可能性' if occ>len(z)/2 else
                       '高さ条件を通る短い枠のy2が窓内の75%値でも低く、足元推定の偏りが残る可能性')
                intervals.append(dict(point=7,player='F1',player_slot='far_rear',start_frame=run[0],end_frame=run[-1],
                                      duration_source_frames=len(run),midpoint_frame=(run[0]+run[-1])//2,
                                      min_depth_m=float(z.court_y.min()*23.77),max_depth_m=float(z.court_y.max()*23.77),
                                      occluded_frames=occ,offscreen_held_frames=held,cause_estimate=cause))
    for col in counts:
        if col.startswith('review2c_'):counts[col]=counts[col].astype(int)
    if write:
        r2.csv('review2_far_depth_counts.csv',counts)
        r2.csv('review2c_speed_reset_intervals.csv',pd.DataFrame(resets,columns=['point','player','player_slot','rally_no','start_frame','rejected_end_frame','reset_frame','rejection_span_source_frames','accepted_depth_m','base_status']))
        r2.csv('review2c_p07_F1_above_intervals.csv',intervals)
        r2.csv('review2c_identity_changes.csv',identity_changes)
    s['review2_depth_counts']=counts.to_dict('records')
    s['review2c_speed_resets']=resets;s['review2c_p07_F1_above_intervals']=intervals
    s['review2c_identity_changed_rows']=len(identity_changes)
    s['rules']['speed_reset']=dict(source_frame_span_strictly_greater_than=30,scope='same player and rally, far and near',
        missing_observations='do not break streak',other_rejections='break streak',retroactive=False,
        status='observed_after_speed_reset',column='speed_reset_frame',smoothing='existing +/-8F median retained')
    s['review2c_checks']=[dict(point=11,source_frame=32970,file='check_review2c_p11_F32970.jpg')]
    if intervals:
        top=max(intervals,key=lambda e:(e['duration_source_frames'],-e['start_frame']))
        f=top['midpoint_frame']
        s['review2c_checks'].append(dict(point=7,source_frame=f,file=f'check_review2c_p07_F{f:05d}.jpg'))
    for check in s['review2c_checks']:
        p=check['point'];b.REVIEW2_CHECKS[p]=sorted(set(b.REVIEW2_CHECKS.get(p,[])+[check['source_frame']]))


def extract_checks(s):
    """Decode the actual MP4 frame, using the saved exact source-frame map."""
    for check in s['review2c_checks']:
        p=check['point'];frame_map=pd.read_csv(OUT/f'video_source_frame_map_p{p:02d}.csv')
        row=frame_map[frame_map.source_frame.eq(check['source_frame'])]
        assert len(row)==1
        output_frame=int(row.iloc[0].output_frame)
        cap=cv2.VideoCapture(str(OUT/f'point_{p:02d}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,output_frame)
        ok,im=cap.read();cap.release();assert ok
        assert cv2.imwrite(str(OUT/check['file']),im,[cv2.IMWRITE_JPEG_QUALITY,95])
        check['output_frame']=output_frame


def write_report(s):
    import review2 as r2
    lines=['## H：review2c','',
           '±15元F、3観測以上、farのみ、元y2より小さくしない条件を維持し、包絡を90→75パーセンタイルへ変更。既存far横座標は固定。速度ゲートはfar・nearとも同一人物・ラリー内で速度棄却列の元F幅が30Fを超えた観測をobserved_after_speed_resetとして採用し、そのFをspeed_reset_frameへ記録する。観測のないFは列を中断せず、高さ棄却・通常採用等は中断する。過去の棄却は遡及採用しない。既存の±8F中央値・30F補間は継続するため、リセット後も直前の全Fが有効になるとは限らない。','',
           '空き名へのゲート免除とreentry_vacant_slotは、ともに最終観測からの差がFPS（60000/1001）元Fを厳密に超える場合だけとした。1秒以内は従来の時間・距離ゲートを適用する。','',
           '### 有効位置と25.3m超過（表示ラリーの全元F）','',
           'review2cの新規有効・欠測化はreview2bとの同じFでの比較。review2との欠測化も別列で記録する。超過の減少だけでは精度改善を示さない。','']
    lines+=r2.md_table(s['review2_depth_counts'],[('point','点'),('player','人物'),('after_above_25_3m','review2超過'),('review2b_above_25_3m','review2b超過'),('review2c_above_25_3m','review2c超過'),('review2b_valid_frames','review2b有効F'),('review2c_valid_frames','review2c有効F'),('review2c_new_valid_frames','新規有効F'),('review2c_lost_valid_frames','失った有効F'),('review2c_lost_valid_frames_vs_review2','review2比で失った有効F')])
    lines+=['','### 速度ゲートのリセット区間','',
            '開始Fは最初の速度棄却観測、棄却区間終了Fはリセット直前の元F。区間内には元観測のないFも含み得る。深さは既存中央値処理後の表示値。','']
    lines+=r2.md_table(s['review2c_speed_resets'],[('point','点'),('player','人物'),('rally_no','ラリー'),('start_frame','開始F'),('rejected_end_frame','棄却区間終了F'),('reset_frame','再初期化F'),('rejection_span_source_frames','開始からの元F幅'),('accepted_depth_m','採用深さm'),('base_status','元の採用種別')])
    if not s['review2c_speed_resets']:lines+=['該当なし。']
    old=pd.read_csv(BACKUP/'review2_reentry_intervals.csv');now=pd.read_csv(OUT/'review2_reentry_intervals.csv')
    lines+=['','### 再入場区間の再集計','',f"review2b：{len(old)}区間・{old.newly_assigned_frames.sum()}新規割当F → review2c：{len(now)}区間・{now.newly_assigned_frames.sum()}新規割当F。人物名が変わった対応行は{s['review2c_identity_changed_rows']}行。reentry表はreview2前との新規割当区間であり、短い欠測を通常ゲートで接続できた区間は含まない。",'']
    lines+=r2.md_table(s['review2_reentries'],[('point','点'),('player','人物'),('track_id','track'),('start_frame','開始F'),('end_frame','終了F'),('newly_assigned_frames','新規割当F数'),('assignment_start_frame','接続開始F')])
    changes=[]
    key=['point','player','track_id','start_frame','end_frame']
    oldkeys={tuple(x[k] for k in key) for x in old.to_dict('records')};newkeys={tuple(x[k] for k in key) for x in now.to_dict('records')}
    for kind,table,other in [('削除',old,newkeys),('追加',now,oldkeys)]:
        changes += [dict(change=kind,**x) for x in table.to_dict('records') if tuple(x[k] for k in key) not in other]
    lines+=['','区間表の差分：','']+r2.md_table(changes,[('change','変化'),('point','点'),('player','人物'),('track_id','track'),('start_frame','開始F'),('end_frame','終了F')])
    versions={name:pd.read_csv(directory/'cleaned_player_positions_p11.csv').set_index(['source_frame','player_slot'])
              for name,directory in [('review2',r2.env.BACKUP),('review2b',BACKUP),('review2c',OUT)]}
    depths=[]
    for f in list(range(32900,33031,8))+[33030]:
        record=dict(frame=f)
        for name,table in versions.items():
            q=table.loc[f,'far_front'];record[name]=f'{q.court_y*23.77:.3f}' if q.status in b.VALID and pd.notna(q.court_y) else '欠測'
        depths.append(record)
    r2.csv('review2c_p11_F2_depth_comparison.csv',depths)
    lines+=['','### 第11点F2：F32900〜33030の深さ','',
            '単位m。8F刻み＋末尾F33030。欠測は配置として有効な座標がないF。','']
    lines+=r2.md_table(depths,[('frame','元F'),('review2','review2'),('review2b','review2b'),('review2c','review2c')])
    for f in (32970,33000,33001):
        q=versions['review2c'].loc[f,'far_front']
        lines+=['',f'F{f}：{q.status} / filter={q.filter_status}、表示深さ {q.court_y*23.77:.3f}m。']
    lines+=['','review2bの実際の欠測区間F32946〜33000（55F）はすべて回復した（指定されたF32948〜33000を含む）。ただしF33001の深さは約25.96mで、ユーザーの映像確認に基づくベースライン付近の目安（約23.6m）との差が残る。この修正は打球直前の欠測を回復するが、足元深さの正確さを保証しない。','']
    lines+=['','### 第7点F1：25.3m超が残る全区間','',
            '原因は枠・遮蔽状態に基づく推定であり、独立した足元真値での確認ではない。75%包絡でも窓内に短い枠が多ければ偏りが残る。','']
    lines+=r2.md_table(s['review2c_p07_F1_above_intervals'],[('start_frame','開始F'),('end_frame','終了F'),('duration_source_frames','長さF'),('min_depth_m','最小m'),('max_depth_m','最大m'),('occluded_frames','下隠れF'),('offscreen_held_frames','画面外保持F'),('cause_estimate','原因の推定')])
    hits=pd.read_csv(OUT/'hitter_auto.csv');truth=pd.read_csv(OUT/'human_confirmed_hitters_p07_11.csv')
    matches=int(hits.hitter.eq(truth.confirmed_hitter).sum())
    previous=pd.read_csv(BACKUP/'hitter_auto.csv')
    lines+=['','### 打者判定と確認画像','',f'打者判定は確認表と{matches}/32一致。review2bとの打者ラベル一致は{int(hits.hitter.eq(previous.hitter).sum())}/32。第7点6打目の名前解釈は従来どおり暫定で、独立評価・未知試合への精度ではない。','']
    lines+=r2.md_table(s['review2c_checks'],[('point','点'),('source_frame','元F'),('file','左右合成の動画フレーム')])
    visual=OUT/'review2c_visual_review.json'
    if visual.exists():
        review=json.loads(visual.read_text(encoding='utf-8'))
        lines+=['','画像の目視確認：','']
        lines += [f'- {name}：{assessment}' for name,assessment in review['assessments'].items()]
    lines+=['','入力のハッシュ保全・全動画の全Fデコード・独立数式検証の結果はreview2_verification.json。commit・pushは行っていない。','']
    original=(BACKUP/'review2_report.md').read_text(encoding='utf-8')
    (OUT/'review2_report.md').write_text(original.rstrip()+'\n\n'+'\n'.join(lines),encoding='utf-8')
    readme=(BACKUP/'README.md').read_text(encoding='utf-8')
    (OUT/'README.md').write_text(readme+'\n\n## review2c（2026-11-19）\n\n現行の5動画と配置CSVはreview2c。far包絡75%、far/near速度棄却30元F超の再初期化、再接続の空き名判定は最終観測から1秒超。詳細・残存誤差・比較表はreview2_report.mdのH節。過去節は当時の結果として保存。\n',encoding='utf-8')
