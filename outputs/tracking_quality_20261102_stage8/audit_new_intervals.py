"""Frozen transfer integrity and backward-direction diagnostics."""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage7'))
import tracklet_pair_tracking as g
g.OUT=ROOT/'tracking_quality_20261102_stage7'


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag
    baseline=json.loads((ROOT.parent/'tracking_quality_20261102_stage7/players_002_global/protocol.json').read_text(encoding='utf-8'))
    folder=g.OUT/f'players_{tag}_global'
    assert json.loads((folder/'protocol.json').read_text(encoding='utf-8'))==baseline
    assert hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest()==baseline['code_sha256']
    summary,event,near,cache=g.candidates(tag)
    rows=g.a.read(folder/'far_pair_path.csv');keyed={};owners=defaultdict(set)
    for r in rows:
        f=int(r['source_frame']);keyed.setdefault(f,{})[r['player_slot']]=r
        matches=[c for c in cache[f-event['frame']] if str(c['row']['track_id'])==r['track_id']]
        assert len(matches)==1
        assert all(str(matches[0]['row'][k])==r[k] for k in ('x1','y1','x2','y2','bottom_center_x','bottom_center_y'))
        if int(r['stable_tracklet']):owners[r['tracklet_key']].add(r['player_slot'])
    assert all(len(slots)==1 for slots in owners.values())
    for rs in keyed.values():assert len({r['track_id'] for r in rs.values()})==len(rs)<=2
    last=max(f for f,rs in keyed.items() if len(rs)==2)
    ids={s:int(r['track_id']) for s,r in keyed[last].items()}
    rev=list(reversed(cache[:last-event['frame']+1]));path,_=g.solve(rev,dict(seed_ids=ids))
    start={};comparisons=[]
    for offset,choices in enumerate(path):
        frame=last-offset
        for i,j in enumerate(choices):
            if j<0:continue
            slot=('far_rear','far_front')[i];r=rev[offset][j]['row']
            if frame==event['frame']:start[slot]=int(r['track_id'])
            old=keyed.get(frame,{}).get(slot)
            if old:comparisons.append(dict(source_frame=frame,player_slot=slot,iou=g.a.box_iou(r,old)))
    g.write(folder/'forward_backward_comparison.csv',comparisons)
    initial={s:event['seed_ids'][s] for s in ('far_rear','far_front')}
    video=folder/'global_review_h264.mp4'
    media=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames','-of','json',str(video)]))['streams'][0]
    assert int(media['nb_frames'])==summary['frames'] and media['codec_name']=='h264' and media['pix_fmt']=='yuv420p'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],capture_output=True,check=True)
    result=dict(tag=tag,settings_match_reviewed_stage7=True,original_boxes_preserved=True,
        stable_tracklet_identity_changes=0,initial_ids=initial,reverse_returned_initial_ids=start,
        returned_same_initial_ids=start==initial,jointly_observed_comparisons=len(comparisons),
        agreeing_boxes_iou_ge_half=sum(r['iou']>=.5 for r in comparisons),
        near_observation_frames={s:sum(any(r['player_slot']==s for r in rs) for rs in near.values()) for s in ('near_front','near_rear')},
        total_frames=summary['frames'],human_review_status='not_reviewed',
        note='Directional stability is not identity accuracy. Near-side tracks reused from temporal initialization stage.',
        media=dict(full_decode='passed',**media))
    (folder/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
