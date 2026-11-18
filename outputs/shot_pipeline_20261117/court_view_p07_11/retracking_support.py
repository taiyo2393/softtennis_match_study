"""Local pose repair and lift diagnostics for the v2 court view."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PIPE=OUT.parent
SLOTS=['near_front','near_rear','far_rear','far_front']


def recompute_pose(tracks,labels,source_frames,load_module,sha,dump):
    """Remeasure both same-side players only for shots with changed boxes."""
    windows=[]
    original={p:pd.read_csv(PIPE/f'tracking/point_{p:02d}/player_court_proxies.csv').set_index(['source_frame','player_slot']) for p in tracks}
    current={p:t.set_index(['source_frame','player_slot']) for p,t in tracks.items()}
    for label in labels.itertuples():
        point=int(label.point)
        a,b=original[point],current[point]
        slots=SLOTS[:2] if label.side=='N' else SLOTS[2:]
        center=int(label.frame)
        changed=[]
        for frame in range(center-15,center+11):
            for slot in slots:
                key=(frame,slot)
                if (key in a.index)!=(key in b.index):
                    changed.append([frame,slot]);continue
                if key in a.index and not np.array_equal(a.loc[key,['x1','y1','x2','y2']].to_numpy(),
                                                          b.loc[key,['x1','y1','x2','y2']].to_numpy()):
                    changed.append([frame,slot])
        if changed:
            windows.append(dict(point=point,side=label.side,slots=slots,center_frame=center,start_frame=center-15,end_frame_inclusive=center+10,
                                changed_box_frame_slots=changed))
    out=OUT/'pose_retrack_supplement'
    out.mkdir(exist_ok=True)
    signature=dict(tracking_sha256={p:sha(PIPE/f'tracking_v2/point_{p:02d}/player_court_proxies.csv') for p in tracks},
                   extract_pose_sha256=sha(PIPE/'pose/extract_pose.py'),
                   combine_sha256=sha(PIPE/'pose/combine_audio_pose.py'),
                   support_sha256=sha(Path(__file__)),windows=windows)
    manifest_path=out/'manifest.json'
    cached=out/'wrist_timeseries_v2.csv'
    if cached.exists() and manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('signature')==signature and manifest.get('status')=='complete':
            return pd.read_csv(cached),manifest
    if not windows:
        manifest=dict(status='no_changed_shots',signature=signature,windows=[])
        dump(manifest_path,manifest)
        return pd.DataFrame(),manifest
    ext=load_module('extract_pose_retrack_readonly',PIPE/'pose/extract_pose.py')
    os.environ['YOLO_CONFIG_DIR']=str(out/'ultralytics_config')
    ext.CACHE=out
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model=YOLO(str(ext.DEFAULT_MODEL))
    manifest=dict(status='running',signature=signature,windows=windows,device='0',
                  model=str(ext.DEFAULT_MODEL),model_sha256=sha(ext.DEFAULT_MODEL),
                  kp_conf=.3,smooth_window=3,
                  scope='all 32 shots checked; both same-side players for changed shot windows, -15..+10F inclusive',
                  fallback_torso_scope='per 26-frame window and player, as existing wrist_timeseries')
    dump(manifest_path,manifest)
    lookup={(p,int(r.source_frame),r.player_slot):r for p,t in tracks.items() for r in t.itertuples()}
    rows=[]
    for window in windows:
        center=window['center_frame'];point=window['point']
        images,pending=[],[]
        def flush():
            if not images:return
            results=model.predict(source=images,device='0',imgsz=384,conf=.25,iou=.7,max_det=10,
                                  rect=False,batch=32,verbose=False,save=False,stream=False,
                                  project=str(out/'predict_runs'),name='pose',exist_ok=True)
            assert len(results)==len(pending)
            for result,row in zip(results,pending):ext.assign_detection(result,row)
            images.clear();pending.clear()
        for frame,rgb in source_frames(window['start_frame'],window['end_frame_inclusive']):
            raw=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            for slot in window['slots']:
                track=lookup.get((point,frame,slot))
                row=ext.base_row(point,f'retrack_{center}',frame,slot,track)
                rows.append(row)
                if track is not None:
                    crop=ext.make_crop(raw,row,.6,.25,384)
                    if crop is not None:
                        images.append(crop);pending.append(row)
                        if len(images)>=32:flush()
        flush()
        print(f'Pose retrack P{point} F{center}: 26 frames x 2 same-side players complete',flush=True)
    pose=pd.DataFrame(rows,columns=ext.FIELDS)
    assert len(pose)==52*len(windows)
    assert not pose.duplicated(['point','source_frame','player_slot']).any(), 'Overlapping shot windows need merging'
    pose.to_csv(out/'pose_v2.csv',index=False,encoding='utf-8-sig')
    combine=load_module('combine_pose_retrack_readonly',PIPE/'pose/combine_audio_pose.py')
    wrist,diagnostics=combine.wrist_timeseries(pose,.3,3)
    wrist.to_csv(cached,index=False,encoding='utf-8-sig')
    diagnostics.to_csv(out/'wrist_diagnostics.csv',index=False,encoding='utf-8-sig')
    manifest.update(status='complete',pose_rows=len(pose),valid_wrist_speed=int(wrist.wrist_speed.notna().sum()))
    dump(manifest_path,manifest)
    return wrist,manifest


def replace_wrist(wrist,extra):
    if not len(extra):return wrist
    keys=set(zip(extra.point,extra.source_frame,extra.player_slot))
    keep=[key not in keys for key in zip(wrist.point,wrist.source_frame,wrist.player_slot)]
    result=pd.concat([wrist.loc[keep],extra],ignore_index=True)
    assert not result.duplicated(['point','source_frame','player_slot']).any()
    return result


def add_lift(hitters,tracks):
    for shot in hitters:
        shot.update(lift_frames=0,max_lift_px=None,jump_or_step_flag=0,
                    lift_status='undecided_hitter',lift_baseline_y2_px=None,
                    lift_baseline_height_px=None,lift_threshold_px=None,
                    lift_impact_valid_frames=0,lift_baseline_valid_frames=0)
        if not shot['hitter_slot']:continue
        source=tracks[shot['point']]
        source=source[source.player_slot==shot['hitter_slot']]
        delta=source.source_frame-shot['frame']
        impact=source[delta.abs()<=20]
        baseline=source[delta.abs().between(20,40)]
        shot.update(lift_impact_valid_frames=len(impact),lift_baseline_valid_frames=len(baseline))
        if baseline.empty or impact.empty:
            shot['lift_status']='missing_boxes';continue
        reference=float(baseline.y2.median())
        height=float((baseline.y2-baseline.y1).median())
        rise=reference-impact.y2
        count=int((rise>=.06*height).sum())
        shot.update(lift_frames=count,max_lift_px=max(0.,float(rise.max())),jump_or_step_flag=int(count>=5),
                    lift_status='evaluated',lift_baseline_y2_px=reference,
                    lift_baseline_height_px=height,lift_threshold_px=.06*height)
    return hitters


def changes(hitters,baseline):
    old={(int(s.point),int(s.frame)):s for s in baseline.itertuples()}
    result=[]
    for shot in hitters:
        previous=old[shot['point'],shot['frame']]
        if previous.hitter!=shot['hitter'] or previous.status!=shot['status']:
            result.append(dict(point=shot['point'],frame=shot['frame'],side=shot['side'],
                               old_hitter=previous.hitter,old_status=previous.status,
                               new_hitter=shot['hitter'],new_status=shot['status']))
    return result


def append_readme(summary,hitters):
    lines=['','## 連続静止区間による除外（2026-11-19）','',
        '各ByteTrack IDの区間内u/v中央値から0.7m以内、u中央値が−0.05未満または1.05超、2秒以上を静止区間とする。連続したはみ出し・欠測は0.2秒以内（59.94fpsでは最大11F）まで。区間の両端は有効な半径内観測。カメラ支持による追加の門は設けない。',
        '時間は両端を含む元フレーム数÷(60000/1001)。存在時間は最初〜最後の観測の時間幅で欠測を含む。静止区間は開始時刻順に最初の2秒窓を探し、毎回中央値を再計算して規則を満たす間だけ右端を延長する。重複しない区間の合計が30%以上ならID全体、30%未満なら区間の前後ceil(0.5×fps)=30Fを除外する。',
        '全ID全体の中央値・90%条件は比較用の旧統計として残すが、今回の除外条件には使わない。人物属性としての審判を認識した結果ではなく、観客等を含む横外静止人物の規則による除外。',
        '再実行: `study/.venv/Scripts/python.exe -B -X utf8 outputs/shot_pipeline_20261117/retrack_excluding_officials.py --stationarity-mode stationary-run --replay-only`。以前の出力はtracking_v2/strict_reference_*に保存。','',
        '全件の枠付き確認画像: [contact sheet](../tracking_v2/official_review/contact_sheet_all.jpg)。拡大用は同フォルダのcontact_sheet_page_01〜04.jpg。全70件の画像・元フレームは[CSV](../tracking_v2/official_candidates.csv)を参照。各IDの最長静止区間の中央コマを使用し、欠測なら最寄りの観測コマを明記する。','',
        '|点|track_id|除外|静止秒/存在秒|静止割合|理由|',
        '|---|---:|---|---:|---:|---|']
    for r in summary.get('official_candidates',[]):
        mode='全体' if r['exclusion_mode']=='whole_track' else '区間（前後0.5秒）'
        lines.append(f'|{r["point"]}|{r["track_id"]}|{mode}|{r["stationary_seconds"]:.3f}/{r["existence_seconds"]:.3f}|{r["stationary_fraction"]:.2%}|横外・半径0.7m・2秒以上、静止区間{len(r["stationary_runs"])}件|')
    lines+=['','## 元の追跡との差','',
        '共通フレーム・同じスロットの生代理点を10.97m×23.77mで換算。欠測の増減は距離差の件数に含めない。除外なし再生と元追跡は全点完全一致。','',
        '|点|スロット|0.5m以上|消失|追加|','|---|---|---:|---:|---:|']
    for r in summary.get('tracking_comparison',[]):
        if r['comparison']=='original_vs_v2':
            lines.append(f'|{r["point"]}|{r["player_slot"]}|{r["position_difference_ge_0_5m"]}|{r["removed_frames"]}|{r["added_frames"]}|')
    lines+=['','## 打球者の再推定と変更','',
        '全32打球について同側2人の−15〜+10Fの枠を元追跡と比較し、変化または欠測増減のあった打球だけ同側2人を再推定。未変化の窓は既存の手首系列を使用。第9点の既存補完も再利用する。短窓の胴長代替尺度は窓内中央値のため、尺度差も判定変更に寄与しうる。',
        f'再推定{len(summary.get("pose_retrack",{}).get("windows",[]))}打球。32打球の内訳: {summary["hitter_status_counts"]}。正解率ではない。','',
        '|点|F|変更前|変更後|','|---|---:|---|---|']
    for r in summary.get('hitter_changes',[]):
        lines.append(f'|{r["point"]}|{r["frame"]}|{r["old_hitter"]} ({r["old_status"]})|{r["new_hitter"]} ({r["new_status"]})|')
    lines+=['','## 前後位置△','',
        '決定した打球者の生枠について、打球±20Fの枠下端y2と20≤|F−打球F|≤40のy2中央値との差を計算。基準区間の枠高さ中央値の6%以上の上昇が5F以上なら△。位置を補正せず、ジャンプと踏み込みは区別しない。未決定は未評価。','',
        f'△は{sum(s["jump_or_step_flag"] for s in hitters)}件。詳細はhitter_auto.csv。',
        '第8点F25353〜25965はretrack_interval_check.jsonと確認画像を参照。横位置uがコート幅内でも縦位置vがベースライン外の選手はいる。区間全体の人物同一性・位置精度を保証するものではない。']
    with (OUT/'README.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
