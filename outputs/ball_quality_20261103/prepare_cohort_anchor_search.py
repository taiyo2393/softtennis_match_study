"""Freeze source-numbered coarse evidence before assigning a new contact bracket."""
import argparse,csv,hashlib,json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
SOURCE=Path(r'C:\Users\ta239\Research\softtennis_match_study\study/data/raw/match1.mp4')
FPS=60000/1001
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);tag=ap.parse_args().tag
    queue=BASE/'cohort_and_baselines/fixed30_processing_queue.csv';row=next(r for r in read(queue) if r['candidate_id'].endswith('_'+tag))
    out=BASE/'cohort_anchor_proposals'/tag;out.mkdir(parents=True,exist_ok=False)
    start=float(row['candidate_start_sec']);end=float(row['candidate_end_sec'])+3
    seconds=np.arange(start,end+.01,.5);frames=[round(t*FPS) for t in seconds]
    dump(out/'search_protocol.json',dict(date_jst='2026-11-10',candidate_id=row['candidate_id'],source_video=str(SOURCE),
        source_clock='legacy_edited_match1',selection='Entire inherited candidate interval plus3seconds;every0.5sec before new contact/validity/tracking decisions.',
        frame_ids=frames,fps=FPS,source_queue_sha256=sha(queue),source_video_bytes=SOURCE.stat().st_size,
        script_sha256=sha(Path(__file__)),scope='AI chronological image search, not independent human service-validity or outcome GT. Do not infer service contact from first tracking initialization.'))
    cap=cv2.VideoCapture(str(SOURCE));cells=[]
    for f in frames:
        cap.set(1,f);ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
        preview=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA);assert cv2.imwrite(str(out/f'F{f}.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,95])
        header=np.zeros((34,640,3),np.uint8);cv2.putText(header,f'{tag} F{f} {f/FPS:.3f}s',(10,25),0,.65,(255,255,255),1,cv2.LINE_AA)
        cells.append(np.vstack([header,cv2.resize(preview,(640,360),interpolation=cv2.INTER_AREA)]))
    cap.release()
    for page,start in enumerate(range(0,len(cells),16),1):
        group=cells[start:start+16]
        while len(group)%4:group.append(np.zeros_like(cells[0]))
        image=np.vstack([np.hstack(group[i:i+4]) for i in range(0,len(group),4)])
        assert cv2.imwrite(str(out/f'coarse_page_{page:02d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,95])
    dump(out/'coarse_image_hashes.json',{p.name:sha(p) for p in out.glob('*.jpg')})
    print(json.dumps(dict(candidate=tag,frames=len(frames),pages=(len(frames)+15)//16),ensure_ascii=False))

if __name__=='__main__':main()
