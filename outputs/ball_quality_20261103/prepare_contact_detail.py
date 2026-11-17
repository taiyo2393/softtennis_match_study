"""Save source-frame contact evidence without person tracking or outcome inference."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'ball_quality_20261110/cohort_anchor_proposals'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--first',type=int,required=True);ap.add_argument('--last',type=int,required=True);ap.add_argument('--step',type=int,default=2);ap.add_argument('--name',default='contact_detail');args=ap.parse_args()
    out=BASE/args.tag;protocol=json.loads((out/'search_protocol.json').read_text(encoding='utf-8'));folder=out/args.name;folder.mkdir()
    frames=list(range(args.first,args.last+1,args.step));cap=cv2.VideoCapture(protocol['source_video']);cells=[]
    for f in frames:
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        cv2.imwrite(str(folder/f'F{f}_raw.jpg'),cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,95])
        crop=im[1000:1900,500:1400];thumb=cv2.resize(crop,(450,450),interpolation=cv2.INTER_AREA)
        header=np.zeros((32,450,3),np.uint8);cv2.putText(header,f'F{f} {f/(60000/1001):.3f}s',(8,24),0,.7,(255,255,255),1,cv2.LINE_AA)
        cells.append(np.vstack([header,thumb]))
    cap.release()
    for page,start in enumerate(range(0,len(cells),16),1):
        group=cells[start:start+16]
        while len(group)%4:group.append(np.zeros_like(cells[0]))
        cv2.imwrite(str(folder/f'page_{page:02d}.jpg'),np.vstack([np.hstack(group[i:i+4]) for i in range(0,len(group),4)]),[cv2.IMWRITE_JPEG_QUALITY,95])
    (folder/'manifest.json').write_text(json.dumps(dict(source_video=protocol['source_video'],frames=frames,native_crop_xyxy=[500,1000,1400,1900],images={p.name:sha(p) for p in folder.glob('*.jpg')},script_sha256=sha(Path(__file__)),independent_GT=False),indent=2),encoding='utf-8')
    print(str(folder))
if __name__=='__main__':main()
