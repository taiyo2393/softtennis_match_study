import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
import gap_pair_tracking as g

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent
g.OUT=ROOT/'experiment/tracking_quality_20261102_stage7'


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag
    folder=g.OUT/f'players_{tag}_global'
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['code_sha256']==hashlib.sha256((ROOT/'gap_pair_tracking.py').read_bytes()).hexdigest()
    summary,event,near,cache=g.candidates(tag)
    rows=g.a.read(folder/'far_pair_path.csv')
    keyed={};owners=defaultdict(set)
    for r in rows:
        f=int(r['source_frame']);keyed.setdefault(f,{})[r['player_slot']]=r
        candidates=[c for c in cache[f-event['frame']] if str(c['row']['track_id'])==r['track_id']]
        assert len(candidates)==1
        assert all(str(candidates[0]['row'][k])==r[k] for k in ('x1','y1','x2','y2','bottom_center_x','bottom_center_y'))
        if int(r['stable_tracklet']):owners[r['tracklet_key']].add(r['player_slot'])
    assert all(len(v)==1 for v in owners.values())
    for rs in keyed.values():assert len({r['track_id'] for r in rs.values()})==len(rs)<=2
    if tag=='003':oldpath=PREVIOUS/'tracking_quality_20261102_stage9/recompute/tracking_quality_20261102_stage7/players_003_global/far_pair_path.csv'
    elif tag in ('007','025'):oldpath=PREVIOUS/'tracking_quality_20261102_stage8/tracking_quality_20261102_stage7'/f'players_{tag}_global/far_pair_path.csv'
    else:oldpath=PREVIOUS/'tracking_quality_20261102_stage7'/f'players_{tag}_global/far_pair_path.csv'
    before=g.a.read(oldpath)
    fields=('source_frame','player_slot','track_id','x1','y1','x2','y2')
    oldset={tuple(r[k] for k in fields) for r in before};newset={tuple(r[k] for k in fields) for r in rows}
    last=max(f for f,rs in keyed.items() if len(rs)==2)
    ids={s:int(r['track_id']) for s,r in keyed[last].items()}
    rev=list(reversed(cache[:last-event['frame']+1]))
    path,_=g.solve(rev,dict(seed_ids=ids))
    start_ids={};comparisons=[]
    for offset,choices in enumerate(path):
        frame=last-offset
        for i,j in enumerate(choices):
            if j<0:continue
            slot=('far_rear','far_front')[i];r=rev[offset][j]['row']
            if frame==event['frame']:start_ids[slot]=int(r['track_id'])
            other=keyed.get(frame,{}).get(slot)
            if other:comparisons.append(dict(source_frame=frame,player_slot=slot,iou=g.a.box_iou(r,other)))
    g.write(folder/'forward_backward_comparison.csv',comparisons)
    video=folder/'global_review_h264.mp4'
    meta=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames','-of','json',str(video)]))['streams'][0]
    assert int(meta['nb_frames'])==summary['frames'] and meta['codec_name']=='h264' and meta['pix_fmt']=='yuv420p'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],capture_output=True,check=True)
    result=dict(tag=tag,stable_tracklet_identity_changes=0,stable_assigned_tracklets=len(owners),
        old_observation_rows=len(before),new_observation_rows=len(rows),
        exact_path_matches_previous_version=oldset==newset,
        shared_rows=len(oldset&newset),removed_rows=len(oldset-newset),added_rows=len(newset-oldset),
        initial_ids={s:event['seed_ids'][s] for s in ('far_rear','far_front')},reverse_returned_initial_ids=start_ids,
        jointly_observed_comparisons=len(comparisons),agreeing_boxes_iou_ge_half=sum(r['iou']>=.5 for r in comparisons),
        warning='Direction consistency and tracklet ownership are not independent identity ground truth.',
        media=dict(full_decode='passed',**meta))
    (folder/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
