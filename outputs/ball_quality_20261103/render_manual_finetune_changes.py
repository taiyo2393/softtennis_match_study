"""Visual evidence for paired audit gains and errors; no new labels requested."""
import json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,sha,cv2,np

OUT=ROOT/'balltrack_manual_finetune_pilot'

def main():
    assert not (OUT/'visual_verification.json').exists()
    changes=read(OUT/'paired_status_changes.csv')
    selected=[r for r in changes if r['role']=='audit']
    assert len(selected)==4
    frames=[];details=[]
    for index,r in enumerate(selected,1):
        cid=r['candidate_id'];t=int(r['local_frame']);source=SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'
        im=cv2.imread(str(source));assert im is not None and im.shape[:2]==(1080,1920)
        truth=r['truth_visibility']=='visible'
        center=[float(r['truth_x_px']),float(r['truth_y_px'])] if truth else [float(r['new_x']),float(r['new_y'])]
        left=max(0,min(1280,round(center[0])-320));top=max(0,min(720,round(center[1])-180))
        canvas=np.full((660,1920,3),22,np.uint8)
        canvas[120:,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
        canvas[120:,960:]=cv2.resize(im[top:top+360,left:left+640],(960,540),interpolation=cv2.INTER_LINEAR)
        def circle(x,y,color):
            cv2.circle(canvas,(round(x/2),120+round(y/2)),8,color,2,cv2.LINE_AA)
            if left<=x<left+640 and top<=y<top+360:
                cv2.circle(canvas,(960+round((x-left)*1.5),120+round((y-top)*1.5)),12,color,2,cv2.LINE_AA)
        if float(r['old_x']) or float(r['old_y']):circle(float(r['old_x']),float(r['old_y']),(255,255,0))
        circle(float(r['new_x']),float(r['new_y']),(240,80,255))
        if truth:
            x,y=float(r['truth_x_px']),float(r['truth_y_px'])
            for px,py in [(round(x/2),120+round(y/2)),(960+round((x-left)*1.5),120+round((y-top)*1.5))]:
                cv2.line(canvas,(px-10,py),(px+10,py),(80,255,100),2)
                cv2.line(canvas,(px,py-10),(px,py+10),(80,255,100),2)
        cv2.putText(canvas,f'Audit {cid[-3:]} local {t}: {r["before"]} -> {r["after"]}',(16,30),0,.77,(245,245,245),1,cv2.LINE_AA)
        cv2.putText(canvas,'Green cross=stored manual center; magenta=selected fine-tuned output; cyan=original output',(16,68),0,.66,(235,235,235),1,cv2.LINE_AA)
        cv2.putText(canvas,'Full view(left0.5x); fixed640x360 crop(right1.5x), uniform aspect. Manual not_visible: no center drawn.' if not truth else 'Full view(left0.5x); fixed640x360 crop(right1.5x), uniform aspect.',(16,105),0,.62,(235,235,235),1,cv2.LINE_AA)
        target=OUT/f'audit_change_{index}_{cid[-3:]}_{t}.jpg';assert cv2.imwrite(str(target),canvas)
        frames.append(canvas);details.append(dict(case=r['sample_id'],before=r['before'],after=r['after'],
            source_image=str(source),source_image_sha256=sha(source),render=str(target.resolve()),
            crop_xywh=[left,top,640,360],right_scale=1.5,source_aspect_preserved=True))
    atlas=OUT/'audit_changes_contact_sheet.jpg';cv2.imwrite(str(atlas),np.concatenate(frames,axis=0))
    report=dict(images=details,source_labels='Existing user-confirmed manual center table, not newly generated labels',
        gains_and_errors_both_included=True,independent_performance_claim=False,
        labels_and_reviewed_outputs_not_modified=True,atlas=str(atlas.resolve()))
    (OUT/'visual_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(images=[r['render'] for r in details],atlas=str(atlas.resolve())),indent=2))

if __name__=='__main__':main()
