"""Apply imported v2 motion geometry to human shot times and inferred receivers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PIPE, module, labels, ignore_large, md_table, dump, sha
import json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent


def prepare(v2, shot, nxt, tracks, protocol):
    r=v2.prepare(shot,nxt,tracks,protocol)
    if not v2.truth(shot.is_serve):
        return r
    # A receiver must belong to the side opposite this server. Consecutive
    # same-side serves cannot supply a receiving contact for this assumption.
    side=r['front_side']
    r.update(receiver_slot='',receiver_basis='receiver_unresolved',
             role_selection_basis='serve_nonreceiver_farthest_net_at_next_contact',
             touch_next=np.nan,touch_reason='receiver_inferred_not_verified')
    if nxt is None or nxt.hitter_side != side:
        r.update(front_slot='',role_issue='serve_next_receiving_contact_missing',
                 receiver_basis='no_next_opponent_contact',role_ambiguous=False)
        return r
    choices=[]
    for slot in v2.SLOTS[side]:
        f=int(nxt.contact_frame)
        t=tracks[slot]
        p=t.loc[f] if f in t.index else None
        if v2.finite_xy(p):
            choices.append((abs(float(p.y_m)-v2.NET),slot))
    if len(choices)!=2:
        r.update(front_slot='',role_issue='serve_receiver_positions_missing',role_ambiguous=False)
        return r
    # Farthest from net, tie resolved by slot name (stable and documented).
    choices.sort(key=lambda x:(-x[0],x[1]))
    receiver=choices[0][1]
    slot=next(s for s in v2.SLOTS[side] if s!=receiver)
    c=int(shot.contact_frame)
    t=tracks[slot]
    p=t.loc[c] if c in t.index else None
    r.update(front_slot=slot,receiver_slot=receiver,receiver_basis='farthest_net_at_next_contact',
             role_issue='',role_ambiguous=False,
             front_net_distance_m=abs(float(p.y_m)-v2.NET) if v2.finite_xy(p) else np.nan,
             role_distance_gap_m=abs(choices[0][0]-choices[1][0]))
    return r


def merge_pose_caches(v2, boxes, prepared):
    """Select pose by actual cached frame ranges, never by legacy rally names."""
    combined, ranges=[],[]
    for point, slots in boxes.items():
        result={}
        paths=sorted((PIPE/'pose/pose_cache').glob(f'point_{point:02d}_rally_*.csv'))
        covered=set()
        for path in paths:
            pose=pd.read_csv(path,usecols=['source_frame'])
            lo,hi=int(pose.source_frame.min()),int(pose.source_frame.max())
            interval=set(range(lo,hi+1))
            assert not covered.intersection(interval), f'Overlapping pose caches: {point}'
            covered.update(interval)
            ranges.append(dict(point=point,file=path.name,start_frame=lo,end_frame_inclusive=hi))
        for slot,t in slots.items():
            a=t.copy()
            a['foot_image_x']=a.bottom_center_x
            a['foot_image_y']=a.bottom_center_y
            a['foot_source']='bbox_cache_absent' if not paths else 'bbox_cache_outside'
            for path in paths:
                old_rally=path.stem.split('_rally_')[1]
                b=prepared[point][old_rally][slot]
                # Retain ankles_missing inside caches (no box imputation there).
                inside=~b.foot_source.str.startswith('bbox_cache_')
                cols=['x_m','y_m','foot_image_x','foot_image_y','foot_source']
                a.loc[b.index[inside],cols]=b.loc[inside,cols]
            result[slot]=a
        combined.append((point,result))
    return dict(combined),pd.DataFrame(ranges)


def main():
    v2=module('human_net_motion_v2',PIPE/'net_motion/v2/classify_net_motion_v2.py')
    shots=pd.read_csv(OUT.parent/'shot_table/shot_table.csv',keep_default_na=False)
    human=labels()
    assert len(shots)==32 and shots.hitter_slot.eq('').all()
    assert shots.contact_frame.tolist()==human.frame.tolist()
    # Do not call v1.load(): it reads the retired labels and old shot table.
    boxes,protocols={},{}
    input_paths=[PIPE/'net_motion/v2/classify_net_motion_v2.py',PIPE/'net_motion/classify_net_motion.py',
                 OUT.parent/'shot_table/shot_table.csv',PIPE/'labels_human/human_labels_20261118.csv']
    for point in range(7,12):
        folder=PIPE/f'tracking/point_{point:02d}'
        protocols[point]=json.loads((folder/'protocol.json').read_text(encoding='utf-8-sig'))
        t=pd.read_csv(folder/'player_court_proxies.csv')
        assert not t.duplicated(['source_frame','player_slot']).any()
        t['x_m'],t['y_m']=t.court_proxy_x*v2.WIDTH,t.court_proxy_y*v2.LENGTH
        boxes[point]={s:g.set_index('source_frame').sort_index() for s,g in t.groupby('player_slot')}
        input_paths += [folder/'player_court_proxies.csv',folder/'protocol.json']
    prepared,checks,coordinate_inputs=v2.load_coordinates(boxes,protocols)
    input_paths+=coordinate_inputs
    ankles,cache_ranges=merge_pose_caches(v2,boxes,prepared)
    cache_ranges.to_csv(OUT/'pose_cache_ranges.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'projection_validation.csv',index=False)
    rows,trajectories=[],[]
    for (point,rally),g in shots.groupby(['point','rally_no'],sort=True):
        g=g.sort_values('contact_frame').reset_index(drop=True)
        for i,shot in g.iterrows():
            nxt=g.iloc[i+1] if i+1<len(g) else None
            base=prepare(v2,shot,nxt,boxes[point],protocols[point])
            for method in ['bbox','ankle']:
                # v2.event_tracks imports the same box proxy bookkeeping.
                tracks=v2.event_tracks(point,str(rally),method,boxes,
                    {point:{str(rally):ankles[point]}})
                for w in v2.WS:
                    for d in v2.DS:
                        r,_=v2.classify_event(base,tracks,protocols[point],w,d,method)
                        rows.append(r)
                        if w==v2.MAIN_W and d==v2.MAIN_D and r['front_slot']:
                            a=tracks[r['front_slot']].reindex(np.arange(int(r['contact_frame'])-v2.PRE_FRAMES,r['end_frame']+1))
                            smooth=v2.smooth_positions(a[['x_m','y_m']].to_numpy(float),a.segment.to_numpy())
                            for j,(f,p) in enumerate(a.iterrows()):
                                trajectories.append(dict(id=r['id'],coordinate_method=method,source_frame=int(f),
                                    front_slot=r['front_slot'],foot_source=p.foot_source,
                                    raw_x_m=p.x_m,raw_y_m=p.y_m,smooth_x_m=smooth[j,0],smooth_y_m=smooth[j,1]))
    motions=pd.DataFrame(rows)
    assert len(motions)==32*2*3*3
    assert not motions.duplicated(['id','coordinate_method','W_sec','D_m']).any()
    motions.to_csv(OUT/'motions.csv',index=False)
    pd.DataFrame(trajectories).to_csv(OUT/'trajectories_main.csv',index=False)
    summary=[]
    for method,g in motions.groupby('coordinate_method'):
        s=v2.v1.summarize(g)
        s['coordinate_method']=method
        s['paper_source']='inherited_v2_PAPER_not_rechecked_against_original'
        summary.append(s)
    summary=pd.concat(summary,ignore_index=True)
    summary.to_csv(OUT/'summary.csv',index=False)
    main=motions[(motions.W_sec==v2.MAIN_W)&(motions.D_m==v2.MAIN_D)].copy()
    main.to_csv(OUT/'motions_main.csv',index=False)
    main_summary=summary[(summary.W_sec==v2.MAIN_W)&(summary.D_m==v2.MAIN_D)]
    main_summary.to_csv(OUT/'summary_main.csv',index=False)
    paper=v2.paper_tables(summary)
    paper.to_csv(OUT/'paper_comparison.csv',index=False)
    # Every sensitivity condition, both scopes, with the available paper values.
    summary[summary.paper_percent.notna()].to_csv(OUT/'paper_comparison_all_conditions.csv',index=False)
    classification=main.groupby(['coordinate_method','classification']).size().rename('n').reset_index()
    fallbacks=main.groupby('coordinate_method').agg(events=('id','size'),
        bbox_fallback_events=('uses_bbox_fallback','sum'),mixed_proxy_events=('mixed_ankle_bbox','sum'),
        bbox_fallback_frames=('bbox_fallback_frames','sum'),ankle_valid_frames=('ankle_valid_frames','sum'))
    role=main[main.coordinate_method=='bbox'][['point','rally_no','contact_frame','is_serve',
        'next_contact_frame','next_hitter_side','front_side','front_slot','receiver_slot',
        'receiver_basis','role_issue','role_ambiguous','exclude_p8_near']]
    role.to_csv(OUT/'role_selection.csv',index=False)
    display=main_summary[(main_summary.serves=='include')&
        main_summary.scope.isin(['all','exclude_p8_near'])&
        main_summary.metric.isin(['availability','presence','complexity','single_3dir','single_8dir','final_3dir','final_8dir'])]
    text=f'''# 本人ラベルによる前衛モーション再集計

本人の without_candidates 32打球、5点・6つのrally_no区間を対象とする。各打球に対する相手側の前衛を追うため、対象イベントは32件（約30件）。
末尾の打球も+Wまで解析する。labels/、旧shot_table、Claude目視参照は使用しない。
打球時刻・side・is_serve・outcome・rally_noは本人の値。打者slot・モーション有無・方向は本人の正解にはない。
したがって以下は自動分類の**構成比**で、モーション認識の精度評価ではない。

## v2から引き継ぐ規則

`../../net_motion/v2/classify_net_motion_v2.py` をimportし、load_coordinates / event_tracks / classify_event / smooth_positions / paper_tables とv1.summarizeを使用。
分類時の終点は min(同じpoint・rally_no内の次の打球, c+round(W×fps))。主条件W=0.4秒、D=0.3m、感度条件W=0.3/0.4/0.5秒、D=0.2/0.3/0.5m。
fps=60000/1001、主条件24F≈0.4004秒。次打球が前衛の打球でも例外なし。区間始点は打球前6Fの生座標中央値（4観測以上必要）。
中心中央値5F→中心平均3Fで平滑化、角度75°以上・各分割4F以上・各変位>=Dで単/複を分類。
欠測20%超、連続欠測2F超、端点欠測、tracking segment境界等は判定不能。定数・方向定義はv2のまま。
3方向と8方向は区切り角度が違うため、8方向の前3区分合計は3方向の前と一致しない場合がある。

## 前衛の選択と仮定

非サーブ: 打球時に相手側2人のうちネットから近い方（箱位置）。2人が揃わない場合は役割不明。
ネット距離差0.5m未満はrole_ambiguousとして記録し、v2同様、自動的には除外しない。
サーブ: 次の打球がレシーブ側の場合、そのフレームでレシーブ側2人のうち**ネットから遠い方をレシーバーと仮定**し、もう1人を前衛に選ぶ。
打者slotの本人正解はないためこの仮定を検証していない。同距離はslot名順。足首条件の役割選択も箱位置で行う。
次が同じ側・次打球なし・次打球時の2人位置不足の場合は判定不能。第9点F27464は次が奥側のサーブF27976のため、このレシーバー仮定を適用できない。
第11点F32755には本人がサーブ印を付けていないため非サーブとして処理。is_serveの補完、ラリー分割の変更をしない。

## 箱と足首

箱下端は既存正規化座標×(10.97,23.77)m。load_coordinatesで箱からの射影再現を全{sum(c['n'] for c in checks)}行でassert。
最大正規化誤差={max(c['max_abs_error_normalized'] for c in checks):.3g}。射影式の一致であり実位置の精度検証ではない。
足首はpose/pose_cache直下の既存CSVだけを使用。旧A/Bと本人rally_noの名前は対応させず、各キャッシュの実フレーム範囲で選ぶ。
左右conf>=0.3は中点、片方だけならその足首、両方欠測は欠測。ファイルなし・範囲外だけ箱fallback。
キャッシュ範囲内の低信頼やslot欠測は箱で埋めない。前6Fから終点までで足首と箱が混在する場合はv2同様判定不能。
各イベントにfoot_source_counts_json / bbox_fallback_frames / ankle_valid_frames / uses_bbox_fallback / mixed_ankle_bboxを保存。
第8点手前は線審追跡の問題があるため、主に解釈するのは **exclude_p8_near**。allも監査用に併記。

{md_table(fallbacks.reset_index())}

## 主条件の分類件数（第8点手前を含む全件）

{md_table(classification)}

## 主条件の割合（サーブ含む）

分母: presenceは判定可能、complexityとfinal方向はモーションあり、single方向は単。判定不能は「なし」にしない。
summary_main.csvにはサーブ除外条件も保存。summary.csvは全18条件（足元2×W3×D3）と全スコープを含む。

{md_table(display[['coordinate_method','scope','metric','category_ja','n','denominator','percent','total_shots','classifiable','unclassifiable']].round(3))}

## 井箟ほかとの比較

{md_table(paper.round(3))}

比較値は既存v2.PAPERから継承。論文原文は今回再照合していない。旧READMEは値の出典を依頼文中の要約としており、特に8方向の論文値は合計101.9%で整合しない。
原表の数値・分母の照合前に論文値を確定した引用として使わない。値を勝手に正規化・修正せず、全条件比較CSVにもそのまま保存。
打球後の短時間区間、代理座標、標本、方向の角度定義は論文の観察手続と同一ではない。意図的移動・フェイントをこの軌跡だけから確認したとはいえない。
対象は同一試合5点の32イベントで、除外・判定不能後の分母はさらに小さい。構成比の差から戦術や一般化性能を主張しない。

出力: motions.csv（全576行）、motions_main.csv、summary.csv、summary_main.csv、paper_comparison.csv（主条件）、
paper_comparison_all_conditions.csv、role_selection.csv、pose_cache_ranges.csv、projection_validation.csv、trajectories_main.csv、input_manifest.json。

実行: `study\\.venv\\Scripts\\python.exe -B -X utf8 outputs/shot_pipeline_20261117/v_human/net_motion/classify_net_motion_on_human.py`
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    dump(OUT/'input_manifest.json',dict(input_sha256={str(p):sha(p) for p in sorted(set(input_paths))},
        imported_classifier=str(Path(v2.__file__)),events=32,motion_rows=len(motions),
        primary_scope='exclude_p8_near',main_W_sec=v2.MAIN_W,main_D_m=v2.MAIN_D,
        receiver_assumption='farthest from net at next opposite-side contact'))
    ignore_large()
    print(classification.to_string(index=False),flush=True)
    print(display[(display.scope=='exclude_p8_near')&display.metric.isin(['presence','complexity','final_3dir'])][
        ['coordinate_method','metric','category_ja','n','denominator','percent','total_shots','classifiable']].to_string(index=False),flush=True)
    print(fallbacks.to_string(),flush=True)


if __name__=='__main__':
    main()
