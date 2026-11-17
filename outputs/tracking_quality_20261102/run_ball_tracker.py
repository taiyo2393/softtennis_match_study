"""Bounded-memory, correctly aligned MS-TrackNetV3 inference on a local video.

All four input frames end AT the output frame. Preserve raw detections;
do not interpolate, extrapolate, or silently remove jumps. Optional median
file reproduces an existing background; otherwise build a model-resolution
median from uniformly sampled frames (recorded as a different preprocessing).
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ball_alignment import SOURCE


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0)
    p.add_argument('--frames',type=int,default=0,help='0 = until end of video')
    p.add_argument('--median',type=Path)
    p.add_argument('--median-samples',type=int,default=120)
    p.add_argument('--threshold',type=float,default=.30)
    p.add_argument('--batch-size',type=int,default=8)
    args = p.parse_args()
    if args.start_frame < 0 or args.frames < 0 or args.batch_size < 1 or args.median_samples < 1:
        p.error('Invalid frame range or batch/median sample count')
    if not 0 < args.threshold < 1:
        p.error('threshold must lie strictly between 0 and 1')
    if (args.output/'balltrack_predictions.csv').exists():
        p.error('Output already exists; select a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    ballroot = SOURCE/'third_party/RacketVision/source/BallTrack'
    sys.path.insert(0,str(ballroot))
    from inference import BallInferencer
    checkpoint = ballroot/'checkpoints/balltrack_best.pth'
    infer = BallInferencer(str(ballroot/'configs/tracknetv3_base.py'),str(checkpoint),
                           device='cuda',thre=args.threshold,batchsize=args.batch_size)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width,height,total = [int(cap.get(k)) for k in
                          (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FRAME_COUNT)]
    end = min(total,args.start_frame+args.frames) if args.frames else total
    if fps <= 0 or not args.start_frame < end:
        raise ValueError('Empty or invalid video interval')
    if args.median:
        median = np.load(args.median)['median']
        median = cv2.resize(median,(infer.width,infer.height))
        background_mode = 'provided_median_resized'
    else:
        indices = np.unique(np.linspace(args.start_frame,end-1,min(args.median_samples,end-args.start_frame),dtype=int))
        samples = []
        # Sequential decoding is cheaper and more reliable than 120 random seeks.
        cap.set(cv2.CAP_PROP_POS_FRAMES,args.start_frame)
        targets = set(int(i) for i in indices)
        for frame_id in range(args.start_frame,end):
            ok,im = cap.read()
            if not ok:
                raise RuntimeError(f'Median pass failed at {frame_id}')
            if frame_id in targets:
                samples.append(cv2.resize(im,(infer.width,infer.height)))
        median = np.median(np.stack(samples),axis=0).astype(np.uint8)
        background_mode = 'median_after_model_resolution_resize'
    np.savez_compressed(args.output/'median.npz',median=median)
    median_tensor = np.moveaxis(median.astype(np.float32)/255.,-1,0)
    decode_start = max(0,args.start_frame-infer.seq_len+1)
    cap.set(cv2.CAP_PROP_POS_FRAMES,decode_start)
    context = deque(maxlen=infer.seq_len)
    batch,indices = [],[]
    fields = ('source_frame','timestamp_sec','x_px','y_px','visible','confidence',
              'position_source','context_padded','input_first_frame','input_last_frame')
    written = 0
    with (args.output/'balltrack_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()

        def flush():
            nonlocal written
            tensor = torch.from_numpy(np.stack(batch)).float().to(infer.device)
            with torch.inference_mode():
                if infer.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        maps,_,_ = infer.model.forward(frames=tensor)
                else:
                    maps,_,_ = infer.model.forward(frames=tensor)
            for frame_id,hm in zip(indices,maps.float().cpu().numpy()):
                x,y,w,h,conf = infer._predict_location(hm[0])
                cx,cy = int((x+w/2)*width/infer.width),int((y+h/2)*height/infer.height)
                writer.writerow(dict(source_frame=frame_id,timestamp_sec=f'{frame_id/fps:.6f}',
                                     x_px=cx,y_px=cy,visible=int(cx != 0 or cy != 0),
                                     confidence=round(conf,4),position_source='model_observed',
                                     context_padded=int(frame_id < infer.seq_len-1),
                                     input_first_frame=max(0,frame_id-infer.seq_len+1),input_last_frame=frame_id))
            written += len(indices)
            batch.clear()
            indices.clear()
            if written % 240 == 0:
                print(f'{written}/{end-args.start_frame}',flush=True)

        for frame_id in range(decode_start,end):
            ok,im = cap.read()
            if not ok:
                raise RuntimeError(f'Inference decode failed at {frame_id}')
            resized = cv2.resize(im,(infer.width,infer.height)).astype(np.float32)/255.
            if not context:
                context.extend([resized]*infer.seq_len)
            else:
                context.append(resized)
            if frame_id < args.start_frame:
                continue
            sequence = np.moveaxis(np.stack(context),-1,1)
            batch.append(np.concatenate([median_tensor[None],sequence],axis=0).reshape(-1,infer.height,infer.width))
            indices.append(frame_id)
            if len(batch) == args.batch_size:
                flush()
        if batch:
            flush()
    cap.release()
    assert written == end-args.start_frame
    metadata = dict(video=str(args.video.resolve()),width=width,height=height,fps=fps,
                    start_frame=args.start_frame,end_frame_exclusive=end,frames=written,
                    threshold=args.threshold,device=str(infer.device),
                    checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    median_source=str(args.median) if args.median else None,
                    background_mode=background_mode,median_samples=args.median_samples if not args.median else None,
                    temporal_alignment='last input frame equals output frame',
                    interpolation=False,warning='Heatmap confidence is not calibrated probability; coordinates use the input video resolution. Online future-free inference is not claimed: median may use the entire interval.')
    (args.output/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
