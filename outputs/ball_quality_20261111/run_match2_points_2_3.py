"""Extend the frozen match2 ball baseline to the next two chronological points."""
import numpy as np
from PIL import Image

import build_044_spatiotemporal_pilot as common
from run_match2_first_point_baseline import run_baseline
from scan_match2_next_points import BASE, OUT


POINTS = [
    dict(point=2, serve_bracket=[13630,13656], end_bracket=[14056,14090],
         median_range=[13560,14080], scoreboard_before=[0,1], scoreboard_after=[1,1],
         scoreboard_frames=[13607,14147]),
    dict(point=3, serve_bracket=[15214,15222], end_bracket=[15506,15522],
         median_range=[15150,15520], scoreboard_before=[1,1], scoreboard_after=[2,1],
         scoreboard_frames=[15167,15587]),
]


def main():
    for name, digest in common.load(BASE/'publication_hashes.json').items():
        assert common.sha(BASE/name)==digest, name
    assert not (OUT/'protocol.json').exists()
    protocol=common.load(BASE/'protocol.json')
    model_root=common.SOURCE/'third_party/RacketVision/source/BallTrack'
    weight=model_root/'checkpoints/balltrack_best.pth'
    assert common.sha(weight)==protocol['weight_sha256']
    references=[]
    for point in POINTS:
        first=point['serve_bracket'][1]+1
        last=point['end_bracket'][0]-1
        point['core_inclusive']=[first,last]
        point['reference_frames']=np.rint(np.linspace(first,last,6)).astype(int).tolist()
        point['median_frames']=np.rint(np.linspace(*point['median_range'],120)).astype(int).tolist()
        assert len(set(point['median_frames']))==120
        references.extend(point['reference_frames'])
    common.dump(OUT/'protocol.json',dict(
        frozen_before_ball_outputs=True,
        source_selection='The next two chronological points after match2 point 1, delimited by raw footage and scoreboard changes.',
        score_order=['内本・内田組','船水・上松組'],
        boundaries='Conservative visual brackets; exact contact and terminal event details are unresolved.',
        points=POINTS,reference_frames_selected_before_inference=references,
        detector=protocol['detector'],weight_sha256=protocol['weight_sha256'],
        reference_labeling='Six uniformly spaced raw frames per point, chosen before inference. Save human visibility and center before displaying predictions.',
    ))
    ref=OUT/'raw_reference_frames'
    ref.mkdir(exist_ok=False)
    results=[]
    for point in POINTS:
        folder=OUT/f'point_{point["point"]:02}'
        first,last=point['core_inclusive']
        for frame in point['reference_frames']:
            with Image.open(folder/'frames'/f'{frame}.jpg') as image:
                image.save(ref/f'F{frame}.png')
        count=run_baseline(weight,model_root,point['median_frames'],output_dir=folder,core=range(first,last+1))
        results.append(dict(point=point['point'],source_frames=last-first+1,candidate_output_frames=count,
                            human_reference_frames_selected=len(point['reference_frames']),human_references_saved=0,
                            baseline_csv_sha256=common.sha(folder/'baseline_unreviewed_candidates.csv')))
        print(results[-1],flush=True)
    common.dump(OUT/'verification.json',dict(points=results,total_reference_frames=len(references),
        ball_weight_verified=True,reference_images={p.name:common.sha(p) for p in sorted(ref.iterdir())}))
    (OUT/'README.md').write_text(
        '# match2 第2点・第3点の球追跡評価\n\n'
        '得点表示が0対1から1対1、続いて2対1へ進む2ラリーを原映像で特定した。'
        'サーブと終端の候補区間を映像から記録し、その内側を第1点と同じTrackNet設定で処理した。\n\n'
        '第2点はF13657〜14055、第3点はF15223〜15505を対象とした。'
        '参照画像は推論前に各ラリー6時点を等間隔で固定した。'
        '球の状態と中心を原画像で確認してから照合する。\n',encoding='utf-8')
    common.dump(OUT/'publication_hashes.json',{p.name:common.sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})


if __name__=='__main__':
    main()
