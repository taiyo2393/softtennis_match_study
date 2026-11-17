"""Compare the frozen gap-recovery settings on match2's first short gap."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import build_044_spatiotemporal_pilot as common
from audit_match2_first_point_gaps import BASE, OUT
from trial_056_crop_reacquisition import components


ANCHORS = (12387, 12390)
MISSING = (12388, 12389)


def main():
    assert not (OUT/'short_gap_protocol.json').exists()
    frozen = common.load(common.HERE/'056_temporal_reacquisition_trial/protocol.json')
    threshold = frozen['candidate_threshold']
    radius = frozen['native_search_radius_px']
    assert ANCHORS[1]-ANCHORS[0] <= frozen['max_gap_source_frames']
    baseline_path = BASE/'baseline_unreviewed_candidates.csv'
    baseline = {int(row['source_frame']):row for row in common.read(baseline_path)}
    assert all(baseline[frame]['candidate_present']=='1' for frame in ANCHORS)
    assert all(baseline[frame]['candidate_present']=='0' for frame in MISSING)
    positions = [np.array([float(baseline[frame][key]) for key in ('x_px','y_px')]) for frame in ANCHORS]
    guides = {frame: positions[0]+(positions[1]-positions[0])*(frame-ANCHORS[0])/(ANCHORS[1]-ANCHORS[0]) for frame in MISSING}
    baseline_protocol = common.load(BASE/'protocol.json')
    model_root = common.SOURCE/'third_party/RacketVision/source/BallTrack'
    weight = model_root/'checkpoints/balltrack_best.pth'
    assert common.sha(weight)==baseline_protocol['weight_sha256']
    common.dump(OUT/'short_gap_protocol.json', dict(
        frozen_before_recovery_outputs=True, targets=list(MISSING), anchors=list(ANCHORS),
        anchor_source='Original detector endpoints judged plausible in the preceding raw-image AI inspection. Human endpoint references not collected.',
        selection='First short gap, raw inspection supports a single incoming flight without visible contact. Exploratory, selected from six baseline gaps.',
        threshold=threshold, search_radius_native_px=radius,
        full_frame_size=[1920,1080], local_crop_size=[960,540],
        selection_rule='Keep a candidate only when exactly one component center is within the 32px search disk.',
        saved_baseline_sha256=common.sha(baseline_path), weight_sha256=common.sha(weight),
    ))
    sys.path.insert(0,str(model_root))
    from inference import BallInferencer

    detector = BallInferencer(str(model_root/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    assert detector.device.type=='cuda'
    detector.model.eval().requires_grad_(False)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    background = np.load(BASE/'median.npz')['median']
    originals = {frame:cv2.imread(str(BASE/'frames'/f'{frame}.jpg')) for frame in range(12384,12391)}
    assert all(image.shape==(1080,1920,3) for image in originals.values())

    def tensor_for(frame, box):
        left, top, right, bottom = box
        images = [background]+[originals[f] for f in range(frame-3,frame+1)]
        resized = [cv2.resize(image[top:bottom,left:right],(512,288)) for image in images]
        return np.moveaxis(np.asarray(resized,dtype=np.float32)/255.,-1,1).reshape(15,288,512)

    def infer(frames, boxes):
        inputs=np.stack([tensor_for(frame,box) for frame,box in zip(frames,boxes,strict=True)])
        tensor=torch.from_numpy(inputs).to(detector.device)
        with torch.inference_mode(),torch.amp.autocast('cuda'):
            output,_,_=detector.model.forward(frames=tensor)
        return output[:,0].float().cpu().numpy()

    batch=list(range(12387,12391))
    full_box=(0,0,1920,1080)
    full=infer(batch,[full_box]*4)
    for frame,hm in zip(batch,full,strict=True):
        x,y,w,h,confidence=detector._predict_location(hm)
        px,py=int((x+w/2)*1920/512),int((y+h/2)*1080/288)
        row=baseline[frame]
        expected=(int(row['x_px']),int(row['y_px'])) if row['candidate_present']=='1' else (0,0)
        assert (px,py)==expected and float(confidence)==float(row['confidence']),frame
    crops=[]
    for frame in MISSING:
        x,y=guides[frame]
        left=max(0,min(960,round(x-480)))
        top=max(0,min(540,round(y-270)))
        crops.append((left,top,left+960,top+540))
    local=infer(MISSING,crops)
    rows=[]
    for method,maps,boxes in [('full_image',full[1:3],[full_box]*2),('local_crop',local,crops)]:
        for frame,hm,box in zip(MISSING,maps,boxes,strict=True):
            left,top,right,bottom=box
            cs=components(hm,threshold,left,top,right-left,bottom-top)
            guide=guides[frame]
            nearby=[c for c in cs if np.hypot(c['x_px']-guide[0],c['y_px']-guide[1])<=radius]
            candidate=nearby[0] if len(nearby)==1 else None
            rows.append(dict(method=method,source_frame=frame,eligible_components=len(nearby),
                candidate_x_px=candidate['x_px'] if candidate else '',candidate_y_px=candidate['y_px'] if candidate else '',
                peak=candidate['peak'] if candidate else '',guide_x=float(guide[0]),guide_y=float(guide[1]),
                crop_box=list(box),identity_status='unconfirmed' if candidate else 'no_unique_candidate'))
    common.write(OUT/'short_gap_comparison.csv',rows)
    np.savez_compressed(OUT/'short_gap_heatmaps.npz',full=full,local=local)
    assert common.sha(baseline_path)==common.load(OUT/'short_gap_protocol.json')['saved_baseline_sha256']
    print(rows,flush=True)


if __name__=='__main__':
    main()
