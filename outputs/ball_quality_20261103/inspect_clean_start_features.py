"""Diagnostic only: discard pose boxes touching their extraction crop edges."""
import json
from pathlib import Path
import numpy as np
from propose_serve_starts import tracks,propose

BASE=Path(__file__).resolve().parent/'serve_start_pilot_tiled_v2'

def clean(data):
    removed=0;frames=[]
    for f in data['frames']:
        people=[]
        for p in f['people']:
            x1,y1,x2,y2=p['box']
            l,t,r,b=(480,360,1536,720) if p['origin']=='far_court' else (0,320,1920,1080)
            if min(x1-l,y1-t,r-x2,b-y2)<=4:
                removed+=1;continue
            people.append(p)
        frames.append(dict(local_frame=f['local_frame'],people=people))
    return dict(frames=frames),removed

if __name__=='__main__':
    for cid in ['016','002']:
        folder=BASE/f'match1_motion_pilot_{cid}'
        data,n=clean(json.loads((folder/'features.json').read_text()))
        fps=json.loads((folder/'protocol.json').read_text())['fps']
        ps,episodes=propose(tracks(data['frames']),np.load(folder/'audio_features.npz'),fps)
        print(cid,'removed',n,'proposals',[(p['person_track_id'],p['raised_first_local_frame'],p['raised_last_local_frame'],p['contact_candidate_first_local_frame'],p['contact_candidate_last_local_frame']) for p in ps])
        for t in [60,600,700,740,850,1000]:
            print(t,[(round((p['box'][0]+p['box'][2])/2),round(p['box'][3]),round(p['box'][3]-p['box'][1]),p['origin']) for p in data['frames'][t//2]['people']])
