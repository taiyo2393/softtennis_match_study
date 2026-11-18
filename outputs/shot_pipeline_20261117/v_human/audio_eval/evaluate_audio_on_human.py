"""Regenerate frozen v1/v2 audio candidates over human-labelled camera scenes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PIPE, module, labels, dump, sha, ignore_large, md_table
import json
import math
import tempfile
from dataclasses import fields
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SEED, REPEATS = 20261118, 1000
KS = (3, 6, 10)


def merge_ranges(ranges):
    merged = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([float(lo), float(hi)])
    return merged


def intervals(human, ui, fps):
    scenes = pd.read_csv(PIPE / 'scenes/scenes.csv', keep_default_na=False)
    rows = []
    for clip in ui['clips']:
        p = int(clip['point'])
        selected = []
        for sc in clip['scenes']:
            source = scenes[scenes.scene_id == sc['id']].iloc[0]
            assert str(source.is_main_camera) == 'True' or sc['id'] == 'p10_s008'
            assert int(source.start_frame) == sc['start_frame']
            assert int(source.end_frame)+1 == sc['end_frame_exclusive']
            selected.append((int(source.start_frame), int(source.end_frame)+1))
        ranges = merge_ranges(selected)
        assert ranges == [[float(clip['interval_start_frame']), float(clip['interval_end_frame_exclusive'])]]
        for j, (lo, hi) in enumerate(ranges):
            rows.append(dict(scope='scene', point=p, interval_id=f'p{p:02d}_scene{j+1}',
                             start_frame=lo, end_frame_exclusive=hi,
                             scene_ids=';'.join(s['id'] for s in clip['scenes'])))
        for rally_no, g in human[human.point == p].groupby('rally_no'):
            lo, hi = float(g.frame.min()) - fps, float(g.frame.max()) + fps
            # Each human rally expanded by exactly one second and clipped to the scene.
            for j, (a, b) in enumerate(ranges):
                left, right = max(a, lo), min(b, hi)
                if right > left:
                    rows.append(dict(scope='rally', point=p, interval_id=f'p{p:02d}_r{rally_no}_{j}',
                                     start_frame=left, end_frame_exclusive=right, scene_ids=''))
    result = pd.DataFrame(rows)
    for scope, g in result.groupby('scope'):
        for p, group in g.groupby('point'):
            group = group.sort_values('start_frame')
            assert all(a <= b for a, b in zip(group.end_frame_exclusive.iloc[:-1], group.start_frame.iloc[1:]))
            for frame in human.loc[human.point == p, 'frame']:
                assert ((group.start_frame <= frame) & (frame < group.end_frame_exclusive)).sum() == 1
    return result


def subset_candidates(candidates, interval):
    return candidates[(candidates.point == interval.point) &
        (candidates.candidate_frame >= interval.start_frame) &
        (candidates.candidate_frame < interval.end_frame_exclusive)].copy()


def window_coverage(ints, candidates, k):
    union, summed, duration = 0., 0., 0.
    for interval in ints.itertuples():
        duration += interval.end_frame_exclusive - interval.start_frame
        windows = [(max(interval.start_frame, c-k), min(interval.end_frame_exclusive, c+k))
                   for c in subset_candidates(candidates, interval).candidate_frame]
        summed += sum(b-a for a,b in windows)
        union += sum(b-a for a,b in merge_ranges(windows))
    return union / duration, summed / duration


def analyse(base, human, candidates, ints, variant, scope, rng):
    matches, nearest, null_rows = [], [], []
    actual = {p: {k:0 for k in KS} for p in range(7,12)}
    random_counts = {p: np.zeros((REPEATS, len(KS)), dtype=int) for p in range(7,12)}
    sources = []
    for interval in ints.itertuples():
        cand = subset_candidates(candidates, interval)
        sources.append(cand)
        lab = human[(human.point == interval.point) & (human.frame >= interval.start_frame) &
                    (human.frame < interval.end_frame_exclusive)].copy()
        lab = lab.rename(columns={'frame':'contact_frame'})
        lab['rally'] = interval.interval_id
        lab['label_id'] = [f'p{r.point:02d}_r{r.rally_no}_f{r.contact_frame}' for r in lab.itertuples()]
        cand['rally'] = interval.interval_id
        frames = lab.contact_frame.to_numpy(int)
        # All same-interval human labels are >20F apart. Thus +/-10F windows
        # are disjoint: any candidate has at most one eligible label, so this
        # vectorised null computation is exactly the 1-to-1 matching count.
        assert len(frames) < 2 or np.diff(np.sort(frames)).min() > 2*max(KS)
        random_frames = rng.integers(math.ceil(interval.start_frame), math.ceil(interval.end_frame_exclusive),
                                     size=(REPEATS, len(cand)))
        for j, k in enumerate(KS):
            mapping = base.greedy_match(lab, cand, k)
            actual[interval.point][k] += len(mapping)
            for l in lab.itertuples():
                pair = mapping.get(l.label_id)
                matches.append(dict(variant=variant,scope=scope,point=l.point,rally_no=l.rally_no,
                    contact_frame=l.contact_frame,tolerance_frames=k,matched=pair is not None,
                    candidate_id=pair[0] if pair else '',delta_frames=pair[1] if pair else np.nan))
            for f in frames:
                random_counts[interval.point][:,j] += np.any(np.abs(random_frames-f) <= k,axis=1)
        for l in lab.itertuples():
            if len(cand):
                # Deterministic tie: earlier candidate.
                c = cand.assign(abs_delta=(cand.candidate_frame-l.contact_frame).abs()).sort_values(
                    ['abs_delta','candidate_frame','candidate_id']).iloc[0]
                delta, cid = int(c.candidate_frame-l.contact_frame), c.candidate_id
            else:
                delta, cid = np.nan, ''
            nearest.append(dict(variant=variant,scope=scope,point=l.point,rally_no=l.rally_no,
                contact_frame=l.contact_frame,nearest_candidate_id=cid,delta_frames=delta,
                abs_delta_frames=abs(delta)))
    all_cand = pd.concat(sources, ignore_index=True)
    rows = []
    for p in [7,8,9,10,11,'all']:
        pi = ints if p == 'all' else ints[ints.point == p]
        pc = all_cand if p == 'all' else all_cand[all_cand.point == p]
        n_labels = len(human) if p == 'all' else int((human.point == p).sum())
        null = sum(random_counts.values()) if p == 'all' else random_counts[p]
        for j,k in enumerate(KS):
            count = sum(a[k] for a in actual.values()) if p == 'all' else actual[p][k]
            ratios = null[:,j] / n_labels
            union, summed = window_coverage(pi, pc, k)
            duration_frames = float((pi.end_frame_exclusive-pi.start_frame).sum())
            rows.append(dict(variant=variant,scope=scope,point=p,tolerance_frames=k,n_labels=n_labels,
                n_matched=count,recall=count/n_labels,n_candidates=len(pc),candidates_per_shot=len(pc)/n_labels,
                duration_frames=duration_frames,duration_sec=duration_frames*base.FRAME_SEC,
                window_union_ratio=union,window_sum_ratio=summed,
                chance_mean=float(ratios.mean()),chance_sd=float(ratios.std(ddof=1)),
                chance_q025=float(np.quantile(ratios,.025)),chance_q975=float(np.quantile(ratios,.975)),
                empirical_p_one_sided=float((1+np.count_nonzero(null[:,j] >= count))/(REPEATS+1))))
            for t in range(REPEATS):
                null_rows.append(dict(variant=variant,scope=scope,point=p,tolerance_frames=k,
                    replicate=t+1,n_matched=int(null[t,j]),recall=float(ratios[t])))
    return rows, matches, nearest, null_rows


def figures(segments, candidates, human, ints, config, evaluation, null, nearest):
    for seg in segments:
        fig, axes = plt.subplots(3,1,figsize=(14,6.5),sharex=True,gridspec_kw={'height_ratios':[3,1,1]})
        mask = (seg.feature_frames >= seg.start_frame) & (seg.feature_frames < seg.end_frame)
        values = seg.features[(config.low_hz,config.high_hz,config.median_sec)]
        axes[0].plot(seg.feature_frames[mask],values[mask],lw=.6,color='#315f83')
        axes[0].axhline(config.threshold,color='grey',ls='--',lw=.8,label='v1 threshold')
        axes[0].set_ylabel('Log-energy rise\n4-12 kHz')
        for ax,variant in zip(axes[1:],['v1','v2']):
            c = candidates[(candidates.point == seg.point) & ((candidates.kept_v2) if variant=='v2' else True)]
            ax.scatter(c.candidate_frame,np.zeros(len(c)),marker='|',s=170,color='#2b8c62',label=variant+' candidate')
            ax.set_yticks([]); ax.set_ylabel(variant)
        h = human[human.point == seg.point]
        for ax in axes:
            for f in h.frame:
                ax.axvline(f,color='#c44536',lw=1.1,alpha=.85)
            for interval in ints[(ints.scope=='rally')&(ints.point==seg.point)].itertuples():
                ax.axvspan(interval.start_frame,interval.end_frame_exclusive,color='#e8ba52',alpha=.12)
            ax.grid(axis='x',alpha=.2)
        axes[0].plot([],[],color='#c44536',label='human contact')
        axes[0].axvspan(-1,-1,color='#e8ba52',alpha=.2,label='rally +/-1 s')
        axes[0].legend(loc='upper right',ncol=3,fontsize=8)
        axes[-1].set_xlim(seg.start_frame,seg.end_frame)
        axes[-1].set_xlabel('Source video frame (60000/1001 fps)')
        axes[0].set_title(f'Match2 point {seg.point:02d}: full main-camera scene')
        fig.tight_layout(); fig.savefig(OUT/f'point_{seg.point:02d}_audio.png',dpi=145); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(variant,k) in zip(axes.ravel(),[(v,k) for v in ['v1','v2'] for k in [6,10]]):
        g=null[(null.variant==variant)&(null.scope=='rally')&(null.point=='all')&(null.tolerance_frames==k)]
        e=evaluation[(evaluation.variant==variant)&(evaluation.scope=='rally')&(evaluation.point=='all')&(evaluation.tolerance_frames==k)].iloc[0]
        ax.hist(g.n_matched,bins=np.arange(-.5,33.5,1),color='#7698ad')
        ax.axvline(e.n_matched,color='#c44536',lw=2,label=f'actual {int(e.n_matched)}/32')
        ax.set(title=f'{variant}, +/-{k}F, p={e.empirical_p_one_sided:.4f}',xlabel='Matched contacts',ylabel='Random trials')
        ax.legend()
    fig.tight_layout();fig.savefig(OUT/'chance_rally.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(scope,variant) in zip(axes.ravel(),[(s,v) for s in ['rally','scene'] for v in ['v1','v2']]):
        delta=nearest[(nearest.scope==scope)&(nearest.variant==variant)].delta_frames.dropna()
        ax.hist(delta,bins=np.arange(delta.min()-.5,delta.max()+1.5),color='#7698ad')
        ax.axvline(0,color='#c44536',lw=1)
        ax.set(title=f'{scope}, {variant}, n={len(delta)}',xlabel='Nearest candidate - human (frames)',ylabel='Contacts')
    fig.tight_layout();fig.savefig(OUT/'nearest_offsets.png',dpi=145);plt.close(fig)


def main():
    sys.path.insert(0,str(PIPE/'audio'))
    base=module('detect_hit_sounds',PIPE/'audio/detect_hit_sounds.py')
    v2=module('human_audio_v2',PIPE/'audio/detect_hit_sounds_v2.py')
    config_data=json.loads((PIPE/'audio/selected_config.json').read_text(encoding='utf-8'))
    config=base.Config(**{f.name:config_data[f.name] for f in fields(base.Config)})
    frozen=json.loads((PIPE/'audio/selected_config_v2.json').read_text(encoding='utf-8'))
    human=labels()
    ui=json.loads((PIPE/'label_ui/data.json').read_text(encoding='utf-8'))
    fps=1/base.FRAME_SEC
    ints=intervals(human,ui,fps)
    ints.to_csv(OUT/'intervals.csv',index=False)
    scene=ints[ints.scope=='scene']
    segments=[base.Segment(int(r.point),'scene',int(r.start_frame),int(r.end_frame_exclusive),pd.DataFrame())
              for r in scene.itertuples()]
    # Only compute the band/window used by the frozen v1 configuration.
    base.BANDS=((config.low_hz,config.high_hz),)
    base.MEDIAN_WINDOWS=(config.median_sec,)
    with tempfile.TemporaryDirectory(prefix='wav_',dir=OUT) as temp:
        base.extract_and_featurize(segments,Path(ui['source_video']),Path(r'C:\ffmpeg\bin\ffmpeg.exe'),Path(temp))
        candidates=v2.add_features(segments,base.detect(segments,config))
        candidates['rule_margin']=v2.gate_margin(candidates,frozen['rule'])
        candidates['kept_v2']=candidates.rule_margin>=0
        candidates.to_csv(OUT/'candidates.csv',index=False)
        pd.concat([pd.DataFrame({'point':s.point,'source_frame_float':s.feature_frames,
            'log_energy_rise':s.features[(config.low_hz,config.high_hz,config.median_sec)]})
            .loc[lambda d:(d.source_frame_float>=s.start_frame)&(d.source_frame_float<s.end_frame)] for s in segments]
            ).to_csv(OUT/'audio_values.csv',index=False)
        rows,matches,nearest,null=[],[],[],[]
        rng=np.random.default_rng(SEED)
        for scope in ['scene','rally']:
            for variant in ['v1','v2']:
                c=candidates if variant=='v1' else candidates[candidates.kept_v2]
                a,b,n,z=analyse(base,human,c,ints[ints.scope==scope],variant,scope,rng)
                rows.extend(a);matches.extend(b);nearest.extend(n);null.extend(z)
        evaluation=pd.DataFrame(rows)
        nearest=pd.DataFrame(nearest)
        null=pd.DataFrame(null)
        evaluation.to_csv(OUT/'evaluation.csv',index=False)
        pd.DataFrame(matches).to_csv(OUT/'label_matches.csv',index=False)
        nearest.to_csv(OUT/'nearest_offsets.csv',index=False)
        null.to_csv(OUT/'chance_distribution.csv',index=False)
        offset_summary=nearest.groupby(['scope','variant']).agg(n=('delta_frames','count'),
            signed_median=('delta_frames','median'),signed_mean=('delta_frames','mean'),
            abs_median=('abs_delta_frames','median'),abs_mean=('abs_delta_frames','mean'),
            abs_p90=('abs_delta_frames',lambda a:a.quantile(.9)),abs_max=('abs_delta_frames','max'))
        offset_summary.to_csv(OUT/'nearest_offset_summary.csv')
        figures(segments,candidates,human,ints,config,evaluation,null,nearest)
    aggregate=evaluation[evaluation.point=='all'].copy()
    display=aggregate[['scope','variant','tolerance_frames','n_matched','n_labels','n_candidates',
                       'candidates_per_shot','recall','window_union_ratio','chance_mean','chance_q025','chance_q975','empirical_p_one_sided']].round(5)
    readme=f'''# 本人32打球による打球音評価

正解は labels_human/human_labels_20261118.csv の without_candidates、32件だけ。labels/ や古い候補CSVは使用しない。
`audio/detect_hit_sounds.py` の extract_and_featurize / detect / greedy_match と、v2 の add_features / gate_margin を import。
v1は selected_config.json（{config.config_id}）、v2は selected_config_v2.json（{frozen['rule_id']}）をそのまま適用し、再探索・時刻補正はしない。
音声は元動画から48kHz monoで再抽出。既存関数の1.5秒パディング、帯域通過、2msエネルギー・1ms刻み、局所中央値を保持。
未使用の帯域・中央値窓の計算だけ省略。WAVはv_human内の一時ディレクトリへ置き、終了時に削除。

## 区間

scenes/scenes.csv の本人ラベルを含む俯瞰場面全体。label_ui/data.json の interval_start_frame / interval_end_frame_exclusive と照合。
第10点 p10_s008（uncertain）も含め、連続するp10_s009と結合。ラベルUIの表示用30F余白は解析区間に加えない。
sceneはこの全体。rallyは同じ点・本人rally_noごとの最初−1秒〜最後＋1秒をsceneでクリップした範囲（60000/1001 fps、±59.94006F）。
第7点は2区間、第9点の2サーブは本人rally_noを保持して1区間。半開区間[start,end)、重複なし。全32件が各scopeに1回ずつ入ることをassert。
候補はscene全体から一度だけ生成し、rally評価時はその候補を切り出す。打球前後で候補を再検出しない。

{md_table(ints.round(5))}

## 指標と偶然水準

±3/6/10Fで同じ点・評価区間内の最近距離優先1対1対応。候補時刻は元の四捨五入による整数Fを使用。
窓割合は連続時間[c−k,c＋k]を区間端で切った**和集合**（重複除去、幅2kF）。window_sum_ratioは重複を含む別指標。
1打球あたり候補数は区間候補数/本人打球数であり、適合率や実測作業時間ではない。

偶然水準は各評価区間の候補数を固定し、区間に入る整数Fへ独立な一様乱数で配置（復元抽出）。
scene/rally、v1/v2ごとに{REPEATS}回、numpy.default_rngのシード{SEED}、ループ順scene→rally、v1→v2。
点・rallyごとの候補密度は維持するが、候補の最小間隔・音の時間依存は保存しない。
本人打球間隔は同一区間内で最短28F>2×10Fなので、各候補が複数正解と競合しない。乱数側の窓内候補有無による計算は1対1対応と同値。
片側の経験的p=(1+乱数対応数が実測以上の回数)/(1000+1)。点別も保存。区間間の合算は各試行の対応数を合計してから計算。
chance_q025/q975は乱数再現率の中央95%範囲で、実測再現率の信頼区間ではない。複数の方法・許容幅のpは未補正。

{md_table(display)}

## 最近候補とのずれ

delta_frames=候補−本人。正は候補が遅い。許容幅による切捨てなし、区間内最近候補を各打球に独立に対応（こちらは1対1制約なし）。
候補が密なら偶然でも小さくなるため、真の打球音源・音響遅延・ラベル精度を証明する値ではない。

{md_table(offset_summary.reset_index().round(4))}

`point_XX_audio.png`: 上段は4–12kHzの対数エネルギー局所差、下段はv1/v2候補。赤縦線は本人打球、淡黄色はrally範囲。横軸は元動画F。
`chance_rally.png`: 主報告の乱数分布と実測。`nearest_offsets.png`: 符号付きずれの分布。
CSV: candidates（再生成全候補とv2保持フラグ・特徴）、audio_values、intervals、evaluation（点別/合算）、label_matches、
chance_distribution（1000回全試行）、nearest_offsets、nearest_offset_summary。

## 解釈上の限界

設定は旧Claudeラベルで選ばれたものを凍結して再評価した。本人ラベルへの再学習はしていないが、同じ試合・既使用区間であり独立テストではない。
一様乱数より高い再現率は打球音源の識別精度や一般化性能を意味しない。32件・5点のみで、打球の独立性も仮定できない。
既存READMEの暫定比較を結論として引き継がず、この全区間再生成の実測と乱数結果を使用する。

実行: `study\\.venv\\Scripts\\python.exe -B -X utf8 outputs/shot_pipeline_20261117/v_human/audio_eval/evaluate_audio_on_human.py`
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    input_paths=[PIPE/'labels_human/human_labels_20261118.csv',PIPE/'labels_human/README.md',PIPE/'scenes/scenes.csv',
                 PIPE/'label_ui/data.json',Path(base.__file__),Path(v2.__file__),PIPE/'audio/selected_config.json',PIPE/'audio/selected_config_v2.json']
    dump(OUT/'run_metadata.json',dict(seed=SEED,repeats=REPEATS,versions={n:__import__(n).__version__ for n in ['numpy','pandas','scipy','matplotlib']},
        input_sha256={p.relative_to(PIPE).as_posix():sha(p) for p in input_paths},
        source_video=ui['source_video'],source_video_bytes=Path(ui['source_video']).stat().st_size,
        temporary_wav_removed=not Path(temp).exists(),frozen_rule=frozen['rule'],retuned=False))
    ignore_large()
    print(display.to_string(index=False),flush=True)
    print(offset_summary.to_string(),flush=True)


if __name__=='__main__':
    main()
