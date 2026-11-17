"""Render selected-player candidate tracks for visual review."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
COLORS={'near_rear':(0,100,255),'near_front':(0,230,255),
        'far_front':(255,120,30),'far_rear':(255,220,0)}

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--mode',choices=('both','bytetrack','botsort_reid'),default='both')
    args=p.parse_args()
    modes=('bytetrack','botsort_reid') if args.mode=='both' else (args.mode,)
    for mode in modes:
        byframe=defaultdict(list)
        with (args.pilot/f'{mode}_selected_four.csv').open(newline='',encoding='utf-8-sig') as file:
            for row in csv.DictReader(file):
                byframe[int(row['source_frame'])].append(row)
        cap=cv2.VideoCapture(str(ROOT/'data/raw/match1.mp4'))
        fps=cap.get(cv2.CAP_PROP_FPS)
        start,end=min(byframe),max(byframe)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        writer=cv2.VideoWriter(str(args.pilot/f'{mode}_selected_four.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720))
        if not writer.isOpened():
            raise RuntimeError('Cannot write preview')
        for frame_idx in range(start,end+1):
            ok,frame=cap.read()
            if not ok:
                break
            h,w=frame.shape[:2]
            canvas=cv2.resize(frame,(1280,720))
            present=set()
            for row in byframe[frame_idx]:
                slot=row['player_slot']; present.add(slot)
                x1,y1,x2,y2=[float(row[k]) for k in ('x1','y1','x2','y2')]
                points=[(int(x1*1280/w),int(y1*720/h)),(int(x2*1280/w),int(y2*720/h))]
                color=COLORS[slot]
                cv2.rectangle(canvas,*points,color,3)
                label=f'{slot} ID{row["track_id"]}'
                cv2.putText(canvas,label,(points[0][0],max(25,points[0][1]-7)),cv2.FONT_HERSHEY_SIMPLEX,.58,color,2)
            missing=','.join(s for s in COLORS if s not in present)
            cv2.rectangle(canvas,(0,0),(800,40),(0,0,0),-1)
            cv2.putText(canvas,f'{mode} F{frame_idx}  missing: {missing or "none"}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            writer.write(canvas)
        cap.release();writer.release()
        print(args.pilot/f'{mode}_selected_four.mp4')

if __name__=='__main__':
    main()
