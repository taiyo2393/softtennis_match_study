"""Freeze source-numbered clips and time-ordered evidence for six provisional prefixes."""
import csv,hashlib,json,shutil,subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
INPUT=ROOT.parent/'ball_quality_20261109/rally_player_prefix_expansion'
OUT=ROOT.parent/'ball_quality_20261110/prefix_phase_audit'
VIDEO=Path(r'C:\Users\ta239\Research\softtennis_match_study\study\data\raw\match1.mp4')
FPS=60000/1001

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def grid(frames,images,meta,path,columns=4,width=640):
    cells=[];height=width*9//16
    for frame in frames:
        im=cv2.resize(images[frame],(width,height),interpolation=cv2.INTER_AREA)
        top=np.zeros((34,width,3),dtype=np.uint8)
        active=int(meta['input_first_source_frame'])<=frame<=int(meta['input_end_source_frame'])
        text=f"{meta['rally_candidate_id'][-3:]} F{frame} {frame/FPS:.3f}s {'INPUT' if active else 'context'}"
        cv2.putText(top,text,(8,24),0,.56,(40,240,255) if active else (255,255,255),1,cv2.LINE_AA)
        cells.append(np.vstack([top,im]))
    while len(cells)%columns:cells.append(np.zeros_like(cells[0]))
    result=np.vstack([np.hstack(cells[i:i+columns]) for i in range(0,len(cells),columns)])
    assert cv2.imwrite(str(path),result,[cv2.IMWRITE_JPEG_QUALITY,95])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Preserve existing evidence.'
    meta=[r for r in read(INPUT/'rally_metadata.csv') if r['inherited_end_bracket_available']=='0']
    assert {r['rally_candidate_id'][-3:] for r in meta}=={'002','003','005','007','025','048'}
    expected='547e5cfd2ed6b0693a83201b1241ba6eb93ab7be8f3f2b135dee2cc48f94258d'
    source_hash=sha(VIDEO);assert source_hash==expected,'Legacy edited source clock required.'
    protected={str(p):sha(p) for p in INPUT.iterdir() if p.is_file()}
    protocol=dict(date_jst='2026-11-10',purpose='AI visual audit of whether fixed early inputs are within the service-led play sequence, not independent human validity/end/position GT.',
        source_video=str(VIDEO),source_video_sha256=source_hash,source_clock='legacy_edited_match1',fps=FPS,
        selection='All six existing prefixes lacking inherited early point-end brackets; no sampling by new detector success or result.',
        context='anchor_first-60 through fixed input_end+180, inclusive. Every source frame in rendered clips; sampled chronological image grids for AI inspection.',
        input_grid='Every fourth source frame over the fixed31-times/stride2 window. Inspecting images is not equivalent to independent continuous-motion judgement.',
        outcomes='Existing human side labels preserved; no score-side relabelling.',
        decisions='Separate visible continued play, cut/reset evidence, ambiguous ball/phase, and validity. No exact contact or point-end certification. No correction of original metadata/features.',
        processing='No new detector, interpolation, model fitting or new human annotation.',protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol);shutil.copy2(Path(__file__),OUT/'executed_preparation_source.py')
    cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened()
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(3840,2160)
    manifest=[]
    for m in meta:
        tag=m['rally_candidate_id'][-3:];folder=OUT/tag;folder.mkdir()
        first=int(m['anchor_first_source_frame'])-60;last=int(m['input_end_source_frame'])+180
        inp=list(range(int(m['input_first_source_frame']),int(m['input_end_source_frame'])+1,4))
        overview=sorted(set(round(f) for f in np.linspace(first,last,24))|{int(m[k]) for k in ['anchor_first_source_frame','anchor_last_source_frame','input_first_source_frame','input_end_source_frame']})
        saved=set(overview+inp);images={};intermediate=folder/'continuous_intermediate.mp4'
        writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(960,576));assert writer.isOpened()
        cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        for f in range(first,last+1):
            ok,im=cap.read();assert ok and int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))==f+1,(tag,f)
            if f in saved:
                images[f]=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA)
                assert cv2.imwrite(str(folder/f'F{f}.jpg'),images[f],[cv2.IMWRITE_JPEG_QUALITY,95])
            thumb=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA);header=np.zeros((36,960,3),dtype=np.uint8)
            active=int(m['input_first_source_frame'])<=f<=int(m['input_end_source_frame'])
            cv2.putText(header,f"{tag} F{f} {f/FPS:.3f}s {'FIXED INPUT' if active else 'context'}",(12,26),0,.7,(0,240,255) if active else (255,255,255),1,cv2.LINE_AA)
            writer.write(np.vstack([header,thumb]))
        writer.release()
        grid(overview,images,m,folder/'context_grid.jpg')
        grid(inp,images,m,folder/'input_grid.jpg')
        clip=folder/'continuous_h264.mp4'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(intermediate),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(clip)],check=True)
        intermediate.unlink()
        cc=cv2.VideoCapture(str(clip));assert int(cc.get(cv2.CAP_PROP_FRAME_COUNT))==last-first+1;cc.release()
        manifest.append(dict(candidate_id=m['rally_candidate_id'],first_source_frame=first,last_source_frame=last,rendered_frames=last-first+1,
            overview_source_frames=overview,input_grid_source_frames=inp,requested_input_range=[int(m['input_first_source_frame']),int(m['input_end_source_frame'])],
            clip=str(clip),clip_sha256=sha(clip),image_hashes={p.name:sha(p) for p in folder.glob('*.jpg')}))
        print(tag,'rendered',last-first+1,'frames',flush=True)
    cap.release()
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'evidence_manifest.json',manifest)
    print('Prepared six source-numbered evidence packages; no phase decisions made.',flush=True)

if __name__=='__main__':main()
