"""Render exact four reviewed exclusions, with separately flagged neighbours."""
import json,subprocess,math
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2,np

OUT=ROOT/'ball_040_object_swap_review'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    assert not (OUT/'render_protocol.json').exists()
    review_dir=OUT/'user_reviewed';verification=load(review_dir/'verification.json');source=Path(verification['experimental_output'])
    assert sha(source)==verification['experimental_sha256'];rows=read(source)
    rejected={int(r['local_frame']):r for r in read(review_dir/'candidate_review.csv')}
    cid='match1_motion_pilot_015';base=SOURCE/'data/balltrack/racketvision/match1'/cid;meta=load(base/'runs/threshold030_median120/run_metadata.json')
    queue=[]
    for t,r in enumerate(rows):
        if t in rejected or not int(r['visible']):continue
        near=[(abs(t-j),math.hypot(float(r['x_px'])-float(q['displayed_x_px']),float(r['y_px'])-float(q['displayed_y_px'])),j)
            for j,q in rejected.items() if abs(t-j)<=2]
        valid=[v for v in near if v[1]<=32]
        if valid:
            d,dist,j=min(valid)
            queue.append(dict(local_frame=t,source_frame=r['source_frame'],x_px=r['x_px'],y_px=r['y_px'],near_rejected_local_frame=j,
                source_frame_distance=d,distance_px=dist,status='unreviewed_neighbour_not_false_label'))
    if queue:write(OUT/'adjacent_review_queue.csv',queue)
    pending={q['local_frame'] for q in queue}
    protocol=dict(status='fixed_before_render',source=str(source),source_sha256=sha(source),rejected_frames=sorted(rejected),
        neighbour_queue='Within2source frames and32pixels of a rejected location, for visual review only; no automatic rejection or training labels',
        pending_frames=sorted(pending),markers='Blue:remaining output,redX:user-rejected location,no ball output; amber:unreviewed neighbour',
        source_dimensions=[1920,1080],render_dimensions=[1920,1180],speed=.5,missing_ball_positions_filled=False)
    dump(OUT/'render_protocol.json',protocol)
    fps=meta['fps'];silent=OUT/'reviewed_040_silent.mp4';video=OUT/'reviewed_040_h264.mp4'
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1180','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for t,r in enumerate(rows):
            image=cv2.imread(str(base/'frames'/f'{t:06d}.jpg'));assert image.shape[:2]==(1080,1920)
            frame=np.full((1180,1920,3),18,np.uint8);frame[100:]=image
            cv2.putText(frame,f'040 | local {t} | source {r["source_frame"]} | {len(rejected)} reviewed false candidates excluded | 0.5x',(12,29),0,.80,(240,240,240),2,cv2.LINE_AA)
            cv2.putText(frame,'BLUE: remaining output',(12,60),0,.70,(255,205,25),2,cv2.LINE_AA)
            cv2.putText(frame,'RED X: rejected; true position unknown',(530,60),0,.70,(60,60,255),2,cv2.LINE_AA)
            if pending:cv2.putText(frame,'AMBER: neighbour still unreviewed',(1330,60),0,.64,(0,190,255),2,cv2.LINE_AA)
            text='REJECTED LOCATION - no replacement ball position confirmed' if t in rejected else 'Neighbour flag is not a false-ball label' if t in pending else 'Other frames not newly reviewed; clip end is not confirmed point-end'
            cv2.putText(frame,text,(12,89),0,.64,(225,225,225),1,cv2.LINE_AA)
            if t in rejected:
                q=rejected[t];p=(round(float(q['displayed_x_px'])),100+round(float(q['displayed_y_px'])))
                cv2.drawMarker(frame,p,(60,60,255),cv2.MARKER_TILTED_CROSS,25,2,cv2.LINE_AA)
                cv2.putText(frame,'NOT BALL',(min(1730,p[0]+23),p[1]-17),0,.65,(60,60,255),2,cv2.LINE_AA)
                assert not int(r['visible'])
            elif int(r['visible']):
                color=(0,190,255) if t in pending else (255,205,25);p=(round(float(r['x_px'])),100+round(float(r['y_px'])))
                cv2.circle(frame,p,13,color,2,cv2.LINE_AA)
                if t in pending:cv2.putText(frame,'UNREVIEWED',(min(1720,p[0]+20),p[1]-17),0,.63,color,2,cv2.LINE_AA)
            if t==464:assert cv2.imwrite(str(OUT/'reviewed_040_preview.jpg'),frame)
            proc.stdin.write(frame.tobytes());proc.stdin.write(frame.tobytes())
        proc.stdin.close();assert proc.wait()==0
    graph='[1:a]atrim=start=0:end='+f'{len(rows)/fps:.9f}'+',asetpts=PTS-STARTPTS,atempo=0.5[a]'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',meta['clip'],'-filter_complex',graph,
        '-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(video)],check=True)
    cap=cv2.VideoCapture(str(video));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1180,1920);n+=1
    cap.release();assert n==1140
    assert sha(source)==verification['experimental_sha256']
    result=dict(video=str(video.resolve()),video_sha256=sha(video),decoded_frames=n,duration_seconds=n/fps,
        rejected_frames=sorted(rejected),unreviewed_neighbour_frames=sorted(pending),
        ball_output_frames=sum(int(r['visible']) for r in rows),recovered_ball_positions=0,interpolated_positions=0,
        curated_changes_are_human_corrections_not_automatic_model_improvement=True)
    dump(OUT/'render_report.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
