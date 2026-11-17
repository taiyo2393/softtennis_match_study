"""Integrity and backward-direction stability check, not independent ID GT."""
import json
import subprocess
from pathlib import Path
import global_pair_tracking as g

OUT = Path(__file__).resolve().parent/'players_048_global'


def main():
    summary,event,near,cache = g.candidates('048')
    rows = g.a.read(OUT/'far_pair_path.csv')
    keyed = {}
    for r in rows:
        frame = int(r['source_frame'])
        candidates = cache[frame-event['frame']]
        matches = [c['row'] for c in candidates if str(c['row']['track_id']) == r['track_id']]
        assert len(matches)==1
        assert all(str(matches[0][k])==r[k] for k in ('x1','y1','x2','y2','bottom_center_x','bottom_center_y'))
        keyed.setdefault(frame,{})[r['player_slot']] = r
    for rs in keyed.values():
        assert len(rs) <= 2
        assert len({r['track_id'] for r in rs.values()}) == len(rs)
    last = max(f for f,rs in keyed.items() if len(rs)==2)
    ids = {s:int(r['track_id']) for s,r in keyed[last].items()}
    rev = list(reversed(cache[:last-event['frame']+1]))
    path,info = g.solve(rev,dict(seed_ids=ids))
    comparisons = []
    reverse_at_start = {}
    for offset,choices in enumerate(path):
        frame = last-offset
        for i,index in enumerate(choices):
            slot = ('far_rear','far_front')[i]
            if index<0:continue
            r = rev[offset][index]['row']
            if frame == event['frame']:
                reverse_at_start[slot] = int(r['track_id'])
            forward = keyed.get(frame,{}).get(slot)
            if forward:
                comparisons.append(dict(source_frame=frame,slot=slot,iou=g.a.box_iou(r,forward)))
    result = dict(reverse_anchor_frame=last,reverse_anchor_ids=ids,
                  reverse_returned_initial_ids=reverse_at_start,
                  original_initial_ids={s:event['seed_ids'][s] for s in ('far_rear','far_front')},
                  jointly_observed_comparisons=len(comparisons),
                  agreeing_boxes_iou_ge_half=sum(r['iou']>=.5 for r in comparisons),
                  note='Reverse anchors are derived from the forward proposal. Directional stability is not independent identity accuracy.')
    g.write(OUT/'forward_backward_comparison.csv',comparisons)
    video = OUT/'global_review_h264.mp4'
    meta=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames','-of','json',str(video)]))['streams'][0]
    assert int(meta['nb_frames'])==summary['frames'] and meta['codec_name']=='h264' and meta['pix_fmt']=='yuv420p'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],capture_output=True,check=True)
    result['media']=dict(full_decode='passed',**meta)
    (OUT/'directional_and_integrity_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
