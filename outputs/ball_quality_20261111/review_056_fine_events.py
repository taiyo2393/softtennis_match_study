"""Native fine grids for AI followup; not human or event GT."""
from pathlib import Path
import cv2,numpy as np
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import OUT,FRAMES,OFFSET

WINDOWS=[('H02',154136,154164,154152,'far_rear'),('H03',154178,154196,154186,'near_front'),
 ('H04',154246,154264,154256,'far_rear'),('H05',154288,154306,154296,'near_front'),
 ('H06',154338,154360,154348,'far_front'),('H07',154458,154482,154470,'near_rear'),
 ('H08',154502,154524,154512,'far_front'),('H09',154632,154656,154644,'near_rear'),
 ('H10',154732,154752,154742,'far_front'),('H11',154770,154794,154780,'near_front'),
 ('H12',154812,154838,154824,'far_rear'),('B01',154114,154128,154120,'far_rear'),
 ('B02',154222,154240,154230,'far_rear'),('B03',154322,154338,154330,'far_front'),
 ('stumble',154586,154610,154596,'near_front')]

def main():
    target=OUT/'fine_review';assert not target.exists();target.mkdir()
    people=b.grouped(b.read(OUT/'player_court_proxies.csv'));manifest=[];plan=[]
    for tag,first,last,reference,slot in WINDOWS:
        pp=next((r for r in people[reference] if r['player_slot']==slot),None)
        if pp is None:
            candidates=[(abs(f-reference),r) for f,rows in people.items() if abs(f-reference)<=12 for r in rows if r['player_slot']==slot]
            pp=min(candidates,key=lambda v:v[0])[1] if candidates else None
        if pp:
            x1,y1,x2,y2=[float(pp[k]) for k in ['x1_px','y1_px','x2_px','y2_px']];cx=(x1+x2)/2;cy=(y1+y2)/2
        else:cx,cy=80,850
        ww,hh=480,320;l=int(np.clip(round(cx-ww/2),0,1920-ww));t=int(np.clip(round(cy-hh/2),0,1080-hh))
        plan.append(dict(id=tag,first=first,last=last,reference=reference,slot_context=slot,roi=[l,t,l+ww,t+hh],not_event_label=True))
        tiles=[]
        for f in range(first,last+1,2):
            path=FRAMES/f'{f-OFFSET:06d}.jpg';im=cv2.imread(str(path));assert im is not None
            h=np.zeros((30,ww,3),np.uint8);cv2.putText(h,f'{tag} raw F{f}',(8,21),0,.53,(255,255,255),1,cv2.LINE_AA)
            tiles.append(np.vstack([h,im[t:t+hh,l:l+ww]]));manifest.append(dict(id=tag,source_frame=f,path=str(path),sha256=b.sha(path)))
        while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(target/(tag+'_raw_grid.jpg')),np.vstack([np.hstack(tiles[j:j+3]) for j in range(0,len(tiles),3)]))
    b.dump(target/'plan.json',dict(scope='AI-selected followup after whole uniform chronology; retain rejected/unclear events; no human labels',windows=plan))
    b.write(target/'source_manifest.csv',manifest)
    print('Saved15raw event grids; no new labels.',flush=True)

if __name__=='__main__':main()
