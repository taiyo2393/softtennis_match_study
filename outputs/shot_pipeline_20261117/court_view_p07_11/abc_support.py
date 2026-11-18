"""Read-only source adapters for head anchors, identity continuity and ball evidence."""
from __future__ import annotations

from collections import defaultdict
from itertools import product
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SLOTS = ['near_front', 'near_rear', 'far_rear', 'far_front']
NAMES = dict(zip(SLOTS, ['N1', 'N2', 'F1', 'F2']))
FPS = 60000 / 1001


def inventory():
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_inventory.csv']
    relative=[str(p.relative_to(ROOT)).replace('\\','/') for p in files]
    result=subprocess.run(['git','check-ignore','-z','--stdin'],input=('\0'.join(relative)+'\0').encode('utf-8'),capture_output=True,cwd=ROOT)
    assert result.returncode in (0,1),result.stderr
    ignored=set(result.stdout.decode('utf-8').split('\0'))
    rows=[];cache_size=0;cache_count=0
    for path,rel in zip(files,relative):
        r=dict(file=str(path.relative_to(OUT)).replace('\\','/'),bytes=path.stat().st_size,ignored=rel in ignored)
        if path.suffix in ('.mp4','.jpg','.png') or r['bytes']>256*1024:assert r['ignored'],rel
        if r['file'].startswith('ball_candidates/'):
            cache_size+=r['bytes'];cache_count+=1
        else:rows.append(r)
    rows.append(dict(file='ball_candidates/ (aggregate)',bytes=cache_size,ignored=True,files=cache_count))
    return rows


def geometry(load_module):
    ref = ROOT/'outputs/ball_quality_20261111'
    common = load_module('projection_readonly', ref/'build_044_spatiotemporal_pilot.py')
    protocol = json.loads((ROOT/'outputs/ball_quality_20261110/match2_camera_transfer_pilot/protocol.json').read_text(encoding='utf-8-sig'))
    # Exported boxes are native 1920x1080; tracking projects 2x canonical boxes.
    # The supplied native homography is exactly H_canonical @ diag(2,2,1).
    H=np.asarray(protocol['image_to_court_homography_canonical']) @ np.diag([2.,2.,1.])
    assert np.array_equal(H,np.asarray(protocol['image_to_court_homography_native']))
    return common.project, H


def mark_occlusion(tracks):
    """Overlap fraction uses far box width; near top must be >=5px above far bottom."""
    rows = tracks.to_dict('records')
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[int(r['source_frame'])].append(r)
    for r in rows:
        r['lower_occluded'] = False
        r['occluder_track_ids'] = ''
        if not r['player_slot'].startswith('far'):
            continue
        occluders = []
        for n in by_frame[int(r['source_frame'])]:
            if not n['player_slot'].startswith('near'):
                continue
            overlap = max(0., min(n['x2'],r['x2'])-max(n['x1'],r['x1']))
            if overlap >= .3*(r['x2']-r['x1']) and n['y1'] <= r['y2']-5:
                occluders.append(str(int(n['track_id'])))
        r['lower_occluded'] = bool(occluders)
        r['occluder_track_ids'] = ';'.join(occluders)
    return pd.DataFrame(rows)


def relink(tracks, segments, clean_player, project, H):
    """Two people per side/rally; known tracks stay bound, new tracks may stay missing."""
    # A precedes B: head-corrected, pre-smoothing projections drive association.
    prepared = []
    all_frames = range(int(tracks.source_frame.min()), int(tracks.source_frame.max())+1)
    for slot in SLOTS:
        rows, _ = clean_player(slot, tracks[tracks.player_slot==slot].to_dict('records'), all_frames)
        prepared.extend(r for r in rows if r['filter_status']!='missing')
    source = pd.DataFrame(prepared)
    # Every source observation remains available for linking, even if speed-filtered.
    source['link_x'] = source.head_court_x.where(source.lower_occluded, source.court_proxy_x)
    source['link_y'] = source.head_court_y.where(source.lower_occluded, source.court_proxy_y)
    boundaries = [s['start_frame'] for s in segments[1:]]
    source['rally_no'] = np.searchsorted(boundaries, source.source_frame, side='right')+segments[0]['rally_no']
    output, mapping, switches, transitions, rejected = [], [], [], [], []
    for rally, rally_rows in source.groupby('rally_no', sort=True):
        for side, slots in [('N', SLOTS[:2]), ('F', SLOTS[2:])]:
            subset = rally_rows[rally_rows.player_slot.isin(slots)]
            bound, last, previous_source = {}, {}, {}
            initialized = False
            for frame, group in subset.groupby('source_frame', sort=True):
                frame = int(frame)
                obs = group.sort_values('player_slot').to_dict('records')
                # Simultaneous source tracks must be distinct observations.
                assert len({int(r['track_id']) for r in obs})==len(obs)
                known = {i:bound[int(r['track_id'])] for i,r in enumerate(obs) if int(r['track_id']) in bound}
                assert len(set(known.values()))==len(known)
                if not initialized:
                    for i,r in enumerate(obs):
                        known[i] = r['player_slot']
                        bound[int(r['track_id'])] = r['player_slot']
                    initialized = True
                unknown = [i for i in range(len(obs)) if i not in known]
                choices = []
                for i in unknown:
                    r = obs[i]
                    eligible = []
                    for slot in slots:
                        if slot in known.values():
                            continue
                        if slot not in last:
                            # First appearance of the second person retains initial slot.
                            if slot==r['player_slot']:
                                eligible.append((slot,0.))
                            continue
                        prev = last[slot]
                        dt = (frame-prev['source_frame'])/FPS
                        dist = math.hypot((r['link_x']-prev['link_x'])*10.97,(r['link_y']-prev['link_y'])*23.77)
                        if 0 < dt <= 1 and dist <= 8*dt+1.5:
                            eligible.append((slot,dist))
                    choices.append([(None,0.),*eligible])
                feasible = []
                for combination in product(*choices):
                    names = [s for s,_ in combination if s is not None]
                    if len(set(names))==len(names):
                        feasible.append(((-len(names),sum(d for _,d in combination)),combination))
                best = min(feasible,key=lambda v:v[0])[1]
                assigned = dict(known)
                for i,(slot,dist) in zip(unknown,best):
                    if slot is not None:
                        assigned[i] = slot
                        bound[int(obs[i]['track_id'])] = slot
                for i,r in enumerate(obs):
                    tid = int(r['track_id'])
                    original = r['player_slot']
                    slot = assigned.get(i)
                    record = dict(point=int(r['point']),rally_no=int(rally),source_frame=frame,
                                  source_slot=original,track_id=tid,player_slot=slot or '',
                                  identity_status='known_track' if i in known else ('linked' if slot else 'unlinked_missing'))
                    mapping.append(record)
                    if slot is None:
                        rejected.append(record)
                        continue
                    if slot in last and int(last[slot]['track_id'])!=tid:
                        prev=last[slot]
                        transitions.append(dict(point=int(r['point']),rally_no=int(rally),frame=frame,player=NAMES[slot],
                                                previous_track=int(prev['track_id']),new_track=tid,
                                                gap_sec=(frame-prev['source_frame'])/FPS,
                                                distance_m=math.hypot((r['link_x']-prev['link_x'])*10.97,(r['link_y']-prev['link_y'])*23.77)))
                    if slot!=original and previous_source.get(original)!=(tid,slot):
                        previous = last.get(original)
                        switches.append(dict(point=int(r['point']),rally_no=int(rally),frame=frame,
                                             original_slot=original,original_name=NAMES[original],fixed_name=NAMES[slot],
                                             previous_track=int(previous['track_id']) if previous else None,new_track=tid))
                    previous_source[original]=(tid,slot)
                    last[slot]=dict(r)
                    r.update(source_slot=original,player_slot=slot,identity_status=record['identity_status'])
                    # Remove intermediate cleaning columns, so final A uses fixed people.
                    output.append(r)
    fixed = pd.DataFrame(output)
    assert not fixed.duplicated(['source_frame','player_slot']).any()
    return fixed, pd.DataFrame(mapping), dict(corrected_switches=switches,track_transitions=transitions,
                                            unlinked_observation_count=len(rejected),unlinked_observations=rejected)


def remap_wrist(wrist, maps):
    """Pose was measured from original slot boxes; move features with those boxes."""
    frames = []
    for point,g in wrist.groupby('point'):
        if int(point) not in maps:
            continue
        m=maps[int(point)][['source_frame','source_slot','player_slot']].rename(columns={'player_slot':'fixed_slot','source_slot':'player_slot'})
        merged=g.merge(m,on=['source_frame','player_slot'],how='inner',validate='one_to_one')
        merged=merged[merged.fixed_slot!=''].copy()
        merged['source_slot']=merged.player_slot
        merged['player_slot']=merged.fixed_slot
        frames.append(merged.drop(columns='fixed_slot'))
    result=pd.concat(frames,ignore_index=True)
    assert not result.duplicated(['point','source_frame','player_slot']).any()
    return result


def combine_ball(hitters, tracks, candidates):
    for shot in hitters:
        p,f=shot['point'],shot['frame']
        slots=SLOTS[:2] if shot['side']=='N' else SLOTS[2:]
        window=candidates[(candidates.point==p)&candidates.source_frame.between(f-3,f+3)&(candidates.candidate_present==1)]
        people=tracks[p].set_index(['source_frame','player_slot'])
        distances=[[],[]]
        paired=[]
        for ball in window.itertuples():
            measured=[]
            widths=[]
            for i,slot in enumerate(slots):
                key=(int(ball.source_frame),slot)
                if key not in people.index:
                    measured.append(None);widths.append(None);continue
                box=people.loc[key]
                w,h=box.x2-box.x1,box.y2-box.y1
                dx=max(box.x1-.3*w-ball.x_px,0.,ball.x_px-box.x2-.3*w)
                dy=max(box.y1-.3*h-ball.y_px,0.,ball.y_px-box.y2)
                dist=math.hypot(dx,dy)
                distances[i].append((dist,int(ball.source_frame),float(w)))
                measured.append(dist);widths.append(float(w))
            paired.append(dict(frame=int(ball.source_frame),x_px=float(ball.x_px),y_px=float(ball.y_px),
                               confidence=float(ball.confidence),distances_px=measured,box_widths_px=widths))
        minima=[min(v) if v else None for v in distances]
        ball_hitter=''
        ball_reason='no_candidates' if window.empty else 'insufficient_boxes_or_ambiguous'
        if all(v is not None for v in minima):
            d=[v[0] for v in minima]
            near=int(d[1]<d[0]);far=1-near
            # Both minima are calculated over exactly the requested +/-3F.
            zero_rule=d[near]==0 and d[far]>=minima[far][2]
            ratio_rule=d[far]>0 and d[near]/d[far]<.5
            if d[near]<d[far] and (zero_rule or ratio_rule):
                ball_hitter=slots[near]
                ball_reason='inside_vs_box_width' if zero_rule else 'distance_ratio_lt_0.5'
        wrist=shot['hitter_slot']
        status='undecided';hitter='';reason='no_decisive_evidence'
        if ball_hitter and wrist:
            if ball_hitter==wrist:
                status,hitter,reason='auto_ball_wrist',wrist,'ball_wrist_agree'
            else:reason='ball_wrist_conflict'
        elif ball_hitter:
            status,hitter,reason='auto_ball',ball_hitter,'ball_only'
        elif wrist and window.empty:
            status,hitter,reason='auto_wrist',wrist,'no_ball_candidates'
        elif wrist:reason='ball_candidates_ambiguous'
        shot.update(wrist_hitter_slot=wrist,wrist_hitter=NAMES.get(wrist,shot['side']+'?'),wrist_reason=shot['reason'],
                    ball_hitter_slot=ball_hitter,ball_hitter=NAMES.get(ball_hitter,shot['side']+'?'),
                    ball_candidate_frames=len(window),ball_reason=ball_reason,
                    ball_player_1_distance_px=minima[0][0] if minima[0] else None,
                    ball_player_2_distance_px=minima[1][0] if minima[1] else None,
                    ball_player_1_best_frame=minima[0][1] if minima[0] else None,
                    ball_player_2_best_frame=minima[1][1] if minima[1] else None,
                    ball_distance_ratio=min(v[0] for v in minima)/max(v[0] for v in minima) if all(v is not None for v in minima) and max(v[0] for v in minima)>0 else None,
                    ball_evidence_json=json.dumps(paired,ensure_ascii=False),
                    status=status,hitter_slot=hitter,hitter=NAMES.get(hitter,shot['side']+'?'),reason=reason)
    return hitters


def write_readme(summary,hitters):
    lines=['# match2 第7〜11点：奥側の前後位置補間・固定済み人物対応と打者判定','',
           'tracking_v2を読み取って再描画した。Bの入力には従来Aの平滑化前位置を維持し、Bで人物を固定した後、最終Aの奥側位置だけを下記の規則で再計算する。Bの対応付け・Cの球と手首による打者判定は変更しない。打球の時刻・側・ラリーは本人ラベル32行。人物名、位置、球候補、打者は自動処理で未確認。N1/N2/F1/F2は各ラリーの初期スロットに従う人物名で、ラリー中は固定し、役割名としては扱わない。','',
           '## 再実行','',
           '```powershell',
           '& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\court_view_p07_11\\build_court_view.py --far-depth-only',
           '& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\court_view_p07_11\\far_depth_revision.py --verify',
           '```','',
           '今回は保存済み球候補をそのまま使用する。生成時の条件は、study/third_party/racketvision-env/Scripts/python.exeからrun_ball_candidates.pyを実行し、run_match2_first_point_baseline.run_baselineを呼び出したもの。チェックポイントのSHA-256も第1点のprotocolと照合済み。BGR・JPEG品質95、512×288、背景中央値120枚、t−3〜t、CUDA AMP、batch=4、閾値0.3、最大成分を使用。背景120枚は各表示区間内から等間隔に選ぶ。最初の3Fの文脈も元動画から取得。元フレーム番号はffmpegの元PTSと全件照合する。','',
           '## A：下が隠れた枠','',
           'far_*だけを変更し、保存済みB/C・手前側をそのまま使用する。同じFの手前枠との横重なりが奥枠幅の10%以上、かつ手前y1<奥y2なら下隠れ。高さの比は遮蔽判定に使わない。','',
           '下隠れでは、そのFの枠中心xと、同一人物・同一ラリーの前後の下隠れでない採用点の画像y2を時間で線形補間したyを使う。区間長は無制限。片側しかなければその側を保持、両側ともなければ欠測。trackingと同じ射影を適用し、速度採用時はobserved_x_depth_interpolated。アンカーは速度採用結果が一致するまで更新し、最終採用点のみを使う。','',
           '下隠れでないFは枠下端を直接使用し、h>1.35×0.73×(y2−415)の大きな枠だけを高さ棄却する。地平線415px、k=0.73は全点で指定値のまま使用し、再推定なし。奥側の高さリセット・80%点基準・頭基準を廃止。従来の速度規則・±8元F中央値平滑化・採用点間30元F以内の通常補間は維持する。','',
           '左の下隠れ枠は実線で名前に「下隠れ」。右の塗り円＋白縁は「横は観測・前後は補間」、白抜き円は全座標の通常補間。表示は中央値平滑化後である。','',
           '点・奥側選手ごとの全status件数はfar_status_counts.csv（表示区間内の全元F／全出力範囲）、通常補間前の棄却内訳はfar_filter_status_counts.csv。前回比較表・範囲外件数は[修正報告](far_depth_revision_report.md)。','']
    lines+=['',
           '## B：人物のつなぎ直し','',
           '従来Aで補正した平滑化前の代理位置から同側2人に対応付ける。この入力と対応付け結果は今回の最終A改訂から切り離して維持する。同じtrack_idはラリー内で同じ人、同時刻の別trackは別人。新trackは既存の人に対して間隔1秒以内、距離8m/s×間隔＋1.5m以下で接続。同時2人なら接続可能数を最大化した上で合計距離を最小にする。接続不可は欠測として記録し、3人目を作らない。ラリー境界で名前を初期化する。姿勢の手首系列も、その枠の人物へ付け替える。初回の2人目が欠測なら初出スロットで初期化する。','',
           '下表は元スロットへ新しいtrackが入り、人物固定によって別名へ戻した区間の開始。前trackはその元スロット名に固定した人物の直前track。全track接続と未接続観測はidentity_summary.json、毎Fの元スロット対応はidentity_map_p*.csv。','',
           '|点|ラリー|F|元の名前|前track→入ってきたtrack|固定後の名前|',
           '|---|---:|---:|---|---|---|']
    for p in range(7,12):
        for r in summary['identity'][p if p in summary['identity'] else str(p)]['corrected_switches']:
            lines.append(f'|{p}|{r["rally_no"]}|{r["frame"]}|{r["original_name"]}|{r["previous_track"]}→{r["new_track"]}|{r["fixed_name"]}|')
    lines+=['','## C：球候補と打者','',
           '球は各ラリー最初の打球−60F〜最後＋90Fを毎F推定。左の橙の小印は「球：自動検出（未確認）」であり、056の確認済みの緑丸とは区別する。球は画像座標のまま扱い、地面へ射影しない。','',
           '打球±3F内の候補から、各人の横に幅30%・上に高さ30%広げた枠への距離を求め、各人の最小値を比較。片方が0かつ他方が他方の元枠幅以上、または近/遠の比が0.5未満なら球の打者。各最小距離を得たF、距離、比、全候補の座標・信頼度・両距離をCSVに保存。','',
           '手首は打球−8〜+4Fの最大速度、両人に有効値があり一意の最大が他方の1.3倍以上。球と手首が一致ならauto_ball_wrist、球のみ決まればauto_ball、球候補がない場合のみ手首だけをauto_wrist。食い違い・球候補が曖昧・双方なしはundecidedとし、双方の候補と理由を残す。','',
           f'球候補がある打球は{summary["ball_candidate_shots"]}/32。判定内訳：{summary["hitter_status_counts"]}。これは検出の利用可能性と自動判定件数であり、精度ではない。','',
           '|点|F|球候補F数|距離1/2 px|球|手首|最終|status|理由|',
           '|---|---:|---:|---|---|---|---|---|---|']
    def num(v):return '欠測' if v is None else f'{v:.2f}'
    for s in hitters:
        lines.append(f'|{s["point"]}|{s["frame"]}|{s["ball_candidate_frames"]}|{num(s["ball_player_1_distance_px"])}/{num(s["ball_player_2_distance_px"])}|{s["ball_hitter"]}|{s["wrist_hitter"]}|{s["hitter"]}|{s["status"]}|{s["reason"]}|')
    lines+=['','### 前回の判定からの変更','',
           '比較元はhitter_auto_before_abc.csv。前回と名前またはstatusが変わったものを全件記載。','',
           '|点|F|変更前|変更後|','|---|---:|---|---|']
    for r in summary['changes_from_previous']:
        lines.append(f'|{r["point"]}|{r["frame"]}|{r["old_hitter"]} ({r["old_status"]})|{r["new_hitter"]} ({r["new_status"]})|')
    targets={(7,23650),(7,23803),(7,23934),(8,25535),(8,25863),(9,28526)}
    lines+=['','### 前回未決定の6打球','',
           '|点|F|球|手首|結果|status|理由|','|---|---:|---|---|---|---|---|']
    for s in hitters:
        if (s['point'],s['frame']) in targets:
            lines.append(f'|{s["point"]}|{s["frame"]}|{s["ball_hitter"]}|{s["wrist_hitter"]}|{s["hitter"]}|{s["status"]}|{s["reason"]}|')
    lines+=['','## 動画・確認画像・検証','',
           '1920×1080、15fps、無音。通常は元映像2Fごと。指定した奇数Fも厳密に収録するため必要な場合だけ1コマ追加する。各ラリー範囲を連結し、対応をvideo_source_frame_map_p*.csvに保存。','',
           '|点|動画|コマ数|秒|全復号|','|---|---|---:|---:|---|']
    for p in range(7,12):
        v=summary['points'][str(p)]['rendering']
        lines.append(f'|{p}|[MP4](point_{p:02d}.mp4)|{v["frames"]}|{v["duration_sec"]:.3f}|{v["full_decode_verified"]}|')
    for name in summary['requested_checks']:
        lines.append(f'\n![{name}]({name})')
    lines+=['','## 限界','',
           '下が隠れている間の前後位置は観測ではなく補間であり、その間の前後への動きは見えない。片側のみの場合は一定値保持である。前後の枠下端自体の誤差も引き継ぐ。固定式による大きな枠の棄却は、2人のまとまりを完全に識別するものではない。位置精度は独立した正解で未検証。人物対応は前回のまま固定し、同一ID内部の人物混同や誤接続は修復しない。','',
           '球候補は自動出力で誤検出を含み、枠に近いだけで接触は証明できない。球と手首の一致も独立した正解ではない。前回未決定の打球が決まっても正解率向上とは結論できない。位置・人物対応・打者の評価には映像による独立した参照ラベルが必要。模式線は打者間の説明用で球軌道・バウンド・3D位置ではない。','',
           'tracking/、tracking_v2/、pose/は読み取り専用で入力SHA-256を前後照合。ball_quality_*への書き込みは行わない。git commitは行わない。256KiB超ファイルと画像・動画のgit除外をgit check-ignoreで検証し、file_inventory.csvに保存する。球キャッシュは一覧をフォルダ合計にまとめるが、全ファイルのgit除外を検査する。','.']
    residual=['','## 今回の確認と集計','',
              '表示区間内の全元フレーム（動画の2F間引きで省いたFも含む）を対象に、最終statusが採用または補間のcourt_yの5%・50%・95%点をmで集計。空間的な切り取りは行わない。移動量は同じラリーの連続する元Fの両方が有効な場合のコート平面ユークリッド距離。欠測やラリー境界はまたがない。単位はm/元フレームで、最大となる遷移の両端Fを記録する。','',
              '集計はdisplayed_position_metrics.csvとfar_depth_revision_report.json、全status件数はfar_status_counts.csvを参照。修正直前との比較・B/Cの16ファイルのSHA-256・手前側の全従来列の一致・変更禁止入力の照合も同reportに保存する。','',
              '第8点F25470のfar_rearのtrack279は、Bの接続結果を維持するため欠測扱いのまま。右図は前後の採用点による補間（白抜き）で表示する。位置誤差の改善を示すには独立した正解位置が必要。']
    if summary.get('displayed_position_metrics'):
        residual += ['', '|点|選手|y5% m|y50% m|y95% m|最大移動 m/元F|元Fの遷移|',
                     '|---|---|---:|---:|---:|---:|---|']
        for r in summary['displayed_position_metrics']:
            residual.append(f'|{r["point"]}|{r["name"]} ({r["player_slot"]})|{r["court_y_p05_m"]:.3f}|{r["court_y_p50_m"]:.3f}|{r["court_y_p95_m"]:.3f}|{r["max_step_m_per_source_frame"]:.6f}|F{r["max_step_from_frame"]}→F{r["max_step_to_frame"]}|')
    (OUT/'README.md').write_text('\n'.join(lines[:-1]+residual)+'\n',encoding='utf-8',newline='\n')
