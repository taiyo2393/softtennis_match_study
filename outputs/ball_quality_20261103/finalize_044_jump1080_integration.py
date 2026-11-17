"""Verify and register the next immutable integration snapshot."""
import json,subprocess
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,sha,cv2

def main():
    out=ROOT/'integrated_044_v3';protocol=load(out/'protocol.json')
    current=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    assert protocol['ball_input_sha256']==current['reviewed_output_sha256']==sha(Path(current['reviewed_output']))
    person=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044']
    assert protocol['player_input_sha256']==person['reviewed_output_sha256']==sha(Path(person['reviewed_output']))
    states={int(r['local_frame']):r for r in read(out/'frame_observation_status.csv')}
    entities=read(out/'observed_entities_1920x1080.csv')
    ball=read(Path(current['reviewed_output']))
    for t in (1081,1082):
        r=next(r for r in entities if int(r['local_frame'])==t and r['entity']=='ball')
        assert r['x_px']==ball[t]['x_px'] and r['y_px']==ball[t]['y_px']
        assert r['review_id']=='20261103_044_jump1080_five_false_two_real'
    for t in (1084,1086,1087,1088):
        assert states[t]['ball_output_present']=='0'
        assert not any(int(r['local_frame'])==t and r['entity']=='ball' for r in entities)
    old_entities=read(ROOT/'integrated_044_v2/observed_entities_1920x1080.csv')
    assert [r for r in entities if r['entity']!='ball']==[r for r in old_entities if r['entity']!='ball']
    final=out/'integrated_review_h264.mp4';fps=59.94005994005994
    # Full snapshot is0.5x. Take1076..1106 and slow it twice again to0.25x.
    start=(1076-739)*2/fps;duration=(1106-1076+1)*2/fps
    local=out/'corrected_1080_excerpt_h264.mp4'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y',
        '-ss',str(start),'-t',str(duration),'-i',str(final),
        '-vf',"setpts=2*(PTS-STARTPTS),drawbox=x=0:y=0:w=1280:h=40:color=0x161616:t=fill,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text=044 corrected 1076-1106 - 0.25x:x=12:y=10:fontsize=23:fontcolor=white",
        '-af','atempo=0.5','-r',str(fps),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(local)],check=True)
    cap=cv2.VideoCapture(str(final))
    for t in (1082,1084):
        cap.set(cv2.CAP_PROP_POS_FRAMES,(t-739)*2);ok,im=cap.read();assert ok
        cv2.imwrite(str(out/f'integrated_local{t}.jpg'),im)
    cap.release()
    cap=cv2.VideoCapture(str(local));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(800,1280);decoded+=1
    cap.release();assert decoded>=120
    probe=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-show_streams','-of','json',str(local)]))
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    rp=ROOT/'reviewed_tracking_bundles.json';registry=load(rp);old=registry['rallies']['match1_rally_window_044']
    assert Path(old['integrated_folder'])==ROOT/'integrated_044_v2'
    registry['rallies']['match1_rally_window_044']=dict(old,ball_input=current['reviewed_output'],
        ball_input_sha256=current['reviewed_output_sha256'],integrated_folder=str(out.resolve()),
        integrated_video=str(final.resolve()),integrated_video_sha256=sha(final),
        frame_status=str((out/'frame_observation_status.csv').resolve()),observed_entities=str((out/'observed_entities_1920x1080.csv').resolve()),
        protocol=str((out/'protocol.json').resolve()),previous_snapshot=old['integrated_folder'])
    rp.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    report=dict(snapshot='integrated_044_v3',reviewed_ball_hash_matches=True,person_entities_identical_to_v2=True,
        positive_components_exported_at_same_frame=True,four_rejected_only_frames_have_no_ball_output=True,
        full_video_decoded_frames=load(out/'verification.json')['video_decoded_frames'],excerpt_decoded_frames=decoded,
        excerpt_audio_present=True,excerpt=str(local.resolve()),independent_accuracy_established=False)
    (out/'final_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
