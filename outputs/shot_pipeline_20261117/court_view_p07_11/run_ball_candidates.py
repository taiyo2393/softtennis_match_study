"""Run the unchanged first-point BallTrack baseline, writing only here."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'outputs/ball_quality_20261111'))
from run_match2_first_point_baseline import run_baseline
import build_court_view as b


def main():
    labels=pd.read_csv(b.PIPE/'labels_human/human_labels_20261118.csv')
    labels=labels[(labels.condition=='without_candidates')&(labels.status=='completed')]
    model=ROOT/'study/third_party/RacketVision/source/BallTrack'
    weight=model/'checkpoints/balltrack_best.pth'
    previous=json.loads((ROOT/'outputs/ball_quality_20261111/match2_ball_first_point/protocol.json').read_text(encoding='utf-8-sig'))
    assert b.sha(weight)==previous['weight_sha256']
    manifest=dict(checkpoint=str(weight),checkpoint_sha256=b.sha(weight),
                  baseline_script_sha256=b.sha(ROOT/'outputs/ball_quality_20261111/run_match2_first_point_baseline.py'),
                  python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                  settings='Unchanged run_baseline: BGR JPEG95, native median120, 512x288, t-3..t, CUDA AMP, threshold0.3, batch4, largest component.',
                  review_status='unreviewed',intervals=[])
    results=[]
    for (point,rally),g in labels.groupby(['point','rally_no'],sort=True):
        first,last=int(g.frame.min())-60,int(g.frame.max())+90
        dest=OUT/f'ball_candidates/point_{point:02d}_rally_{rally}'
        frames=dest/'frames';frames.mkdir(parents=True,exist_ok=True)
        med=np.rint(np.linspace(first,last,120)).astype(int).tolist()
        signature=dict(first=first,last=last,median_frames=med,weight_sha256=b.sha(weight),
                       baseline_script_sha256=manifest['baseline_script_sha256'],video=str(b.VIDEO),
                       source_decoder='ffmpeg original PTS, exact frame checks, JPEG95')
        done=dest/'signature.json'
        cache=dest/'baseline_unreviewed_candidates.csv'
        if not (done.exists() and cache.exists() and json.loads(done.read_text())==signature):
            for f,rgb in b.source_frames(first-3,last):
                assert cv2.imwrite(str(frames/f'{f}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
            run_baseline(weight,model,med,output_dir=dest,core=range(first,last+1))
            b.dump(done,signature)
        data=pd.read_csv(cache)
        assert data.source_frame.tolist()==list(range(first,last+1))
        data.insert(0,'rally_no',int(rally));data.insert(0,'point',int(point))
        results.append(data)
        manifest['intervals'].append(dict(point=int(point),rally_no=int(rally),first=first,last=last,
                                         frames=len(data),candidate_frames=int(data.candidate_present.sum()),
                                         median_frames=med,candidates_sha256=b.sha(cache)))
        print(f'Ball p{point} r{rally}: {int(data.candidate_present.sum())}/{len(data)} frames',flush=True)
    all_rows=pd.concat(results,ignore_index=True)
    all_rows.to_csv(OUT/'ball_candidates.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    b.dump(OUT/'ball_manifest.json',manifest)


if __name__=='__main__':
    main()
