"""All excluded ByteTrack IDs: one annotated source frame each and complete sheets."""
import json,math,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
sys.dont_write_bytecode=True
PIPE=Path(__file__).resolve().parent
sys.path.insert(0,str(PIPE/'court_view_p07_11'))
import build_court_view as view
OUT=PIPE/'tracking_v2/official_review';OUT.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',24)
small=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',18)

def main():
    report=json.loads((PIPE/'tracking_v2/verification.json').read_text(encoding='utf-8'))
    records=report['excluded_candidates'];tiles=[];index=[]
    for point in range(7,12):
        with (PIPE/f'tracking_v2/point_{point:02d}/candidate_cache.pkl').open('rb') as f:cache=pickle.load(f)
        candidates={(f['source_frame'],r['track_id']):r for f in cache['frames'] for r in f['candidates']}
        original=pd.read_csv(PIPE/f'tracking/point_{point:02d}/player_court_proxies.csv')
        for stat in [r for r in records if r['point']==point]:
            run=max(stat['stationary_runs'],key=lambda r:r['duration_frames'])
            center=(run['first_frame']+run['last_frame'])//2
            frame=min((f for f,t in candidates if t==stat['track_id'] and run['first_frame']<=f<=run['last_frame']),key=lambda f:(abs(f-center),f))
            box=candidates[frame,stat['track_id']]
            decoded=list(view.source_frames(frame,frame))
            rgb=decoded[0][1]
            im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
            # Detection cache is canonical 3840x2160; source display is 1920x1080.
            rect=[box[k]/2 for k in ['x1','y1','x2','y2']]
            draw.rectangle(rect,outline=(255,50,40),width=5)
            title=f'P{point:02d} ID{stat["track_id"]} F{frame}  {stat["exclusion_mode"]}  stationary={stat["stationary_fraction"]:.1%}'
            draw.rectangle((0,0,1920,42),fill=(10,20,30));draw.text((10,4),title,font=font,fill='white')
            filename=f'p{point:02d}_id{stat["track_id"]}_F{frame}.jpg'
            im.save(OUT/filename,quality=91)
            tile=Image.new('RGB',(640,440),(12,22,32));tile.paste(im.resize((640,360)),(0,40))
            # Enlarged crop on each thumbnail makes even distant people reviewable.
            x1,y1,x2,y2=rect;pad=max(15,(x2-x1)*.25)
            crop=Image.fromarray(rgb).crop((max(0,int(x1-pad)),max(0,int(y1-pad)),min(1920,int(x2+pad)),min(1080,int(y2+pad))))
            crop.thumbnail((135,240));tile.paste(crop,(640-crop.width,400-crop.height))
            d=ImageDraw.Draw(tile);d.text((6,3),title,font=small,fill='white')
            d.text((6,406),f'run F{run["first_frame"]}-{run["last_frame"]}  {stat["stationary_seconds"]:.2f}/{stat["existence_seconds"]:.2f}s',font=small,fill='white')
            tiles.append(tile)
            assigned=original[original.track_id==stat['track_id']]
            index.append(dict(point=point,track_id=stat['track_id'],exclusion_mode=stat['exclusion_mode'],
                stationary_seconds=stat['stationary_seconds'],existence_seconds=stat['existence_seconds'],
                stationary_fraction=stat['stationary_fraction'],reason=stat['reason'],
                stationary_runs=json.dumps(stat['stationary_runs']),exclusion_ranges=json.dumps(stat['exclusion_ranges']),
                review_image=filename,interval_center_frame=center,shown_frame=frame,
                original_assigned_frames=len(assigned),original_slots=','.join(assigned.player_slot.unique())))
            print(title,flush=True)
    sheet=Image.new('RGB',(4*640,math.ceil(len(tiles)/4)*440),(12,22,32))
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*640,(i//4)*440))
    sheet.save(OUT/'contact_sheet_all.jpg',quality=92)
    for page,start in enumerate(range(0,len(tiles),20),1):
        chunk=tiles[start:start+20]
        sub=Image.new('RGB',(2560,math.ceil(len(chunk)/4)*440),(12,22,32))
        for i,tile in enumerate(chunk):sub.paste(tile,((i%4)*640,(i//4)*440))
        sub.save(OUT/f'contact_sheet_page_{page:02d}.jpg',quality=92)
    pd.DataFrame(index).to_csv(PIPE/'tracking_v2/official_candidates.csv',index=False,encoding='utf-8-sig')
    view.dump(OUT/'review_manifest.json',dict(excluded_tracks=len(records),review_images=len(index),
        all_included=True,central_frame_exact_count=sum(r['shown_frame']==r['interval_center_frame'] for r in index),
        originally_assigned=[r for r in index if r['original_assigned_frames']],
        note='Rule-based exclusions include spectators/outside people; review does not establish an official role.'))
    print('Complete:',len(index),'images',flush=True)
if __name__=='__main__':main()
