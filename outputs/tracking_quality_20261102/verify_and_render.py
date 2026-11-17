"""Check assignment/timing contracts and render aligned ball comparisons."""
import itertools
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ball_alignment import SOURCE, OUT, read, window
from four_player_association import linear_sum_assignment, joint_assignment


def test_contracts():
    rng = np.random.default_rng(20261102)
    for _ in range(50):
        costs = rng.normal(size=(4,6))
        i,j = linear_sum_assignment(costs)
        actual = costs[i,j].sum()
        expected = min(sum(costs[r,c] for r,c in enumerate(p))
                       for p in itertools.permutations(range(6),4))
        assert abs(actual-expected) < 1e-9
        assert len(set(j)) == 4
    # Equal alternatives are intentionally withheld, not assigned arbitrarily.
    assert joint_assignment(np.zeros((4,4))) == []
    assert joint_assignment(np.empty((4,0))) == []
    certain = np.full((4,4),1e6)
    np.fill_diagonal(certain,0)
    assert len(joint_assignment(certain)) == 4
    for t in range(200):
        assert window(t) == window(t+1,legacy=True)
        assert window(t)[-1] == t
    # No duplicate identity or box assignment and no synthesized positions.
    for path in OUT.glob('players_*/selected_four.csv'):
        grouped = defaultdict(list)
        for r in read(path):
            assert r['position_source'] == 'detected_box'
            assert r['identity_status'] == 'candidate_not_ground_truth'
            grouped[int(r['source_frame'])].append(r)
        for rs in grouped.values():
            assert len({r['player_slot'] for r in rs}) == len(rs)
            assert len({r['track_id'] for r in rs}) == len(rs)
            assert len(rs) <= 4
    print('Contracts passed: exact assignment vs brute force, ambiguity/missing handling, frame mapping, output uniqueness.',flush=True)


def verify_cached_equivalence():
    """Numerically check fresh corrected inference against shifted saved output."""
    import sys
    import torch
    ballroot = SOURCE/'third_party/RacketVision/source/BallTrack'
    sys.path.insert(0,str(ballroot))
    from inference import BallInferencer
    infer = BallInferencer(str(ballroot/'configs/tracknetv3_base.py'),
                           str(ballroot/'checkpoints/balltrack_best.pth'),
                           device='cuda',thre=.30,batchsize=3)
    checks = []
    for cid in ('match1_motion_pilot_001','match1_motion_pilot_002'):
        base = SOURCE/'data/balltrack/racketvision/match1'/cid
        saved = {int(r['local_frame']):r for r in read(OUT/'ball_aligned'/cid/'balltrack_predictions.csv')}
        median = np.load(base/'runs/threshold030_median120/median.npz')['median']
        median = cv2.resize(median,(infer.width,infer.height))/255.
        tensors = []
        targets = [16,112,230]
        for t in targets:
            ims = [cv2.imread(str(base/'frames'/f'{i:06d}.jpg')) for i in window(t)]
            frames = np.array([cv2.resize(im,(infer.width,infer.height)) for im in ims],dtype=np.float32)/255.
            data = np.concatenate([np.moveaxis(median,-1,0)[None],np.moveaxis(frames,-1,1)],axis=0)
            tensors.append(data.reshape(-1,infer.height,infer.width))
        tensor = torch.from_numpy(np.array(tensors)).float().to(infer.device)
        with torch.inference_mode():
            if infer.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    maps,_,_ = infer.model.forward(frames=tensor)
            else:
                maps,_,_ = infer.model.forward(frames=tensor)
        maps = maps.float().cpu().numpy()
        for t,hm in zip(targets,maps):
            x,y,w,h,conf = infer._predict_location(hm[0])
            cx,cy = int((x+w/2)*1920/infer.width),int((y+h/2)*1080/infer.height)
            old = saved[t]
            delta = float(np.hypot(cx-float(old['x_px']),cy-float(old['y_px'])))
            assert delta <= 1.5,(cid,t,delta)
            assert int(cx != 0 or cy != 0) == int(old['visible'])
            assert abs(conf-float(old['confidence'])) < .015,(cid,t,conf,old['confidence'])
            checks.append(dict(candidate_id=cid,local_frame=t,coordinate_delta_px=delta,
                               confidence_delta=abs(conf-float(old['confidence']))))
    (OUT/'ball_fresh_inference_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Fresh inference agrees with six shifted cached predictions.',flush=True)


def render():
    for tag,cid,start in [('001','match1_motion_pilot_001',0),('007','match1_motion_pilot_002',700)]:
        base = SOURCE/'data/balltrack/racketvision/match1'/cid
        legacy = read(base/'runs/threshold030_median120/balltrack_predictions.csv')
        aligned = read(OUT/'ball_aligned'/cid/'balltrack_predictions.csv')
        meta = json.loads((base/'runs/threshold030_median120/run_metadata.json').read_text(encoding='utf-8'))
        cap = cv2.VideoCapture(meta['clip'])
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        temp = OUT/f'ball_{tag}_intermediate.mp4'
        writer = cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),meta['fps'],(1920,1080))
        assert writer.isOpened()
        for i in range(start,start+240):
            ok,im = cap.read()
            assert ok
            for r,col,radius in ((legacy[i],(220,70,220),15),(aligned[i],(255,255,0),9)):
                if int(r['visible']):
                    point = (round(float(r['x_px'])),round(float(r['y_px'])))
                    cv2.circle(im,point,radius,col,2)
            cv2.rectangle(im,(0,1020),(1920,1080),(0,0,0),-1)
            cv2.putText(im,f'{tag} F{aligned[i]["source_frame"]} | CYAN: corrected timing / MAGENTA: legacy timing',
                        (15,1048),0,.65,(255,255,255),1)
            cv2.putText(im,'Same model, same threshold 0.30. No interpolation. Circles are model candidates.',
                        (15,1073),0,.60,(255,255,255),1)
            writer.write(im)
        writer.release()
        cap.release()
        result = OUT/f'ball_{tag}_timing_comparison_h264.mp4'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y',
                        '-i',str(temp),'-ss',str(start/meta['fps']),'-t',str(240/meta['fps']),
                        '-i',meta['clip'],'-map','0:v:0','-map','1:a:0?',
                        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                        '-c:a','aac','-movflags','+faststart',str(result)],check=True)
        print(f'Rendered {result.name}',flush=True)


if __name__ == '__main__':
    test_contracts()
    verify_cached_equivalence()
    render()
