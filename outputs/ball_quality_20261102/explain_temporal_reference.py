"""Expose the exact reference selected by the frozen temporal guard."""
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from temporal_consensus_guard import ROOT,OUT,SOURCE,BASE,read,camera_features,camera_mask,sha

DEST=OUT/'reference_audit_20261103'


def explain(rows,t,camera=True):
    cid=rows[0]['candidate_id'];mask=camera_mask(camera_features(cid),camera);segment=np.cumsum(mask.astype(int))
    ts=np.array([int(r['local_frame']) for r in rows if int(r['visible']) and not mask[int(r['local_frame'])] and
                 abs(int(r['local_frame'])-t)<=30 and segment[int(r['local_frame'])]==segment[t]],int)
    xy=np.array([[float(rows[f]['x_px']),float(rows[f]['y_px'])] for f in ts])
    aa,bb=np.triu_indices(len(ts),1);keep=ts[bb]-ts[aa]>=3;aa,bb=aa[keep],bb[keep]
    v=(xy[bb]-xy[aa])/(ts[bb]-ts[aa])[:,None]
    at=xy[aa]+v*(t-ts[aa])[:,None]
    errors=np.linalg.norm(xy[None,:,:]-(at[:,None,:]+v[:,None,:]*(ts[None,:,None]-t)),axis=2)
    inliers=errors<=16;counts=inliers.sum(1)
    spans=np.where(inliers,ts[None,:],-10**9).max(1)-np.where(inliers,ts[None,:],10**9).min(1)
    before=(inliers&(ts[None,:]<t)).sum(1);after=(inliers&(ts[None,:]>t)).sum(1)
    qualified=(counts>=6)&(spans>=12)&(before>=2)&(after>=2)
    assert qualified.any()
    best=np.flatnonzero(qualified)[np.argmax(counts[qualified])];chosen=inliers[best]
    own=np.array([float(rows[t]['x_px']),float(rows[t]['y_px'])])
    supported=[dict(local_frame=int(f),x_px=float(p[0]),y_px=float(p[1]),residual_px=float(e))
               for f,p,e in zip(ts[chosen],xy[chosen],errors[best,chosen])]
    return dict(candidate_id=cid,local_frame=t,raw_xy=own.tolist(),reference_prediction_xy=at[best].tolist(),
                candidate_reference_error_px=float(np.linalg.norm(own-at[best])),
                reference_velocity_px_per_frame=v[best].tolist(),reference_speed_px_per_frame=float(np.linalg.norm(v[best])),
                reference_model_pair_frames=[int(ts[aa[best]]),int(ts[bb[best]])],
                reference_inlier_count=int(counts[best]),reference_span_frames=int(spans[best]),
                reference_model_displacement_px=float(np.linalg.norm(v[best])*spans[best]),
                median_reference_residual_px=float(np.median(errors[best,chosen])),support_points=supported)


def media(r):
    cid=r['candidate_id'];t=r['local_frame']
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    canvas=Image.new('RGB',(1280,850),'#171b24');d=ImageDraw.Draw(canvas)
    d.text((8,5),f"{cid[-3:]} frame {t} | cyan: retained raw candidate | orange: fitted reference",font=font,fill='white')
    d.text((8,30),f"reference speed {r['reference_speed_px_per_frame']:.4f} px/frame | {r['reference_inlier_count']} support points",font=font,fill='white')
    im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
    full=Image.fromarray(cv2.cvtColor(cv2.resize(im,(960,540)),cv2.COLOR_BGR2RGB))
    fd=ImageDraw.Draw(full)
    for (x,y),color in ((r['raw_xy'],'#00ddff'),(r['reference_prediction_xy'],'#ffaa00')):
        x/=2;y/=2;fd.ellipse((x-10,y-10,x+10,y+10),outline=color,width=2)
    canvas.paste(full,(0,65))
    support=r['support_points'];sampled=[support[k] for k in sorted(set([0,len(support)//3,2*len(support)//3,len(support)-1]))]
    for i,p in enumerate(sampled):
        f=p['local_frame'];im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{f:06d}.jpg'));assert im is not None
        x=max(0,min(1600,round(p['x_px'])-160));y=max(0,min(900,round(p['y_px'])-90))
        canvas.paste(Image.fromarray(cv2.cvtColor(im[y:y+180,x:x+320],cv2.COLOR_BGR2RGB)),(320*i,645))
        d.text((320*i+4,617),f"reference frame {f}",font=font,fill='white')
    canvas.save(DEST/f"{cid[-3:]}_{t}_reference.jpg",quality=95)


def main():
    DEST.mkdir(exist_ok=True);reports=[]
    for suffix,t in [('002',318),('014',626),('014',630),('018',208)]:
        cid=f'match1_motion_pilot_{suffix}'
        glove=ROOT/'glove_guard/continuous_predictions'/cid/'balltrack_predictions.csv'
        path=glove if glove.exists() else BASE/cid/'balltrack_predictions.csv'
        r=explain(read(path),t);r['input_sha256']=sha(path);reports.append(r);media(r)
    (DEST/'reference_reports.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='support_points'} for r in reports],indent=2))


if __name__=='__main__':main()
