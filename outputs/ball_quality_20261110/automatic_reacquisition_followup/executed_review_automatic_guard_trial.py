"""Report frozen event probes and draw follow-up control evidence, without GT claims."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110';TRIAL=BASE/'automatic_reacquisition_trial_v2';FOLLOW=BASE/'automatic_reacquisition_followup'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def index(rows):
    result=defaultdict(dict)
    for r in rows:result[int(r['source_frame'])][r['player_slot']]=r
    return result
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    protocol=load(TRIAL/'protocol.json');out=FOLLOW/'control_image_review';out.mkdir();cap=cv2.VideoCapture(protocol['source_video']);reports=[];manifest=[]
    slots=['near_front','near_rear'];colors=[(80,255,80),(255,190,40)]
    for ev in protocol['control_switches']:
        tag=ev['tag'];s=ev['slot'];f=ev['frame'];new=str(ev['new_id']);baseline=read(TRIAL/f'case_{tag}/baseline/selected_four.csv');candidate=read(FOLLOW/f'case_{tag}/selected_four.csv');people=index(candidate)
        matches=[int(r['source_frame']) for r in candidate if r['player_slot']==s and r['track_id']==new and int(r['source_frame'])>=f];first=min(matches) if matches else None
        old_matches=[r for r in baseline if r['player_slot']==s and r['track_id']==new and int(r['source_frame'])>=f]
        report=dict(ev,baseline_observations_after_switch=len(old_matches),first_followup_matching_frame=first,additional_delay_frames=None if first is None else first-f,within_original13frame_budget=first is not None and first-f<=13,AI_evidence_status='pending_image_inspection',independent_human_GT=False);reports.append(report)
        trace=next(r for r in read(TRIAL/f'case_{tag}/baseline/near_transition_trace.csv') if int(r['source_frame'])==f and r['player_slot']==s);plan=next(c for c in protocol['cases'] if c['tag']==tag);before=int(trace['old_observed_frame']);frames=[max(plan['first'],before-12),before,f,f+7,f+13,min(plan['last'],f+30)]
        cells=[]
        for frame in frames:
            cap.set(1,frame);ok,im=cap.read();assert ok and round(cap.get(1))==frame+1
            for slot in slots:
                r=people[frame].get(slot)
                if not r:continue
                b=[round(float(r[k])) for k in ['x1','y1','x2','y2']];i=slots.index(slot);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),colors[i],6);cv2.putText(im,['N1','N2'][i]+' raw'+r['track_id'],(b[0],max(35,b[1]-12)),0,1.2,colors[i],3)
            thumb=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA);head=np.zeros((34,960,3),np.uint8);cv2.putText(head,f'{tag} followup F{frame} target present:{s in people[frame]}',(10,25),0,.76,(255,255,255),1);cells.append(np.vstack([head,thumb]))
        file=out/f'{tag}_{s}_{f}.jpg';cv2.imwrite(str(file),np.vstack([np.hstack(cells[:3]),np.hstack(cells[3:])]))
        manifest.append(dict(control=ev,frames=frames,image_file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    cap.release();dump(out/'manifest.json',manifest)
    probe=[]
    for mode in ['baseline','confirm_new_id','confirm_new_id_opponent_negatives','followup']:
        rows=read(FOLLOW/'case_011/selected_four.csv') if mode=='followup' else read(TRIAL/f'case_011/{mode}/selected_four.csv');people=index(rows)
        probe.append(dict(mode=mode,known_failure_probes=[dict(source_frame=f,near_rear_raw_id=people[f].get('near_rear',{}).get('track_id')) for f in protocol['primary_failure_probe_frames']],N2_raw409_assignment_rows=sum(r['player_slot']=='near_rear' and r['track_id']=='409' for r in rows),raw_ID_counts_are_not_identity_accuracy=True))
    def keyset(rows):return {tuple(r[k] for k in ['source_frame','player_slot','track_id','x1','y1','x2','y2']) for r in rows if r['player_slot'].startswith('near_')}
    stable=keyset(read(TRIAL/'case_005/baseline/selected_four.csv'))==keyset(read(FOLLOW/'case_005/selected_four.csv'))
    dump(FOLLOW/'event_comparison.json',dict(controls=reports,known011_failure_probes=probe,stable005_near_assignments_exactly_unchanged=stable,original_guardBC_decision='Reject:003 valid sustained switch lost44 frames;025 edge observation also not retained.',followup_decision='Experimental candidate only. Three sustained switches recover;025 one-frame edge switch not retained. Needs image inspection and independent evaluation.',independent_identity_accuracy_established=False));print(json.dumps(dict(controls=reports,stable005=stable,probes=probe),ensure_ascii=False))
if __name__=='__main__':main()
