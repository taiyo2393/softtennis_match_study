"""Inspect all near-pair clothing distances in retained initialization cases."""
import itertools,json
from pathlib import Path
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from court_initialization_guard import court_point,eligible,old

OUT=BASE/'pair_initialization_development'
def load(p):return json.loads(p.read_text(encoding='utf-8'))

def plans():
    trial=BASE/'court_initialization_trial';p=load(trial/'protocol.json');answer=[]
    for entry in p['cases']:
        tag=entry['tag'];e=dict(entry)
        e['cache']=entry['cache'] or str(trial/f'case_{tag}')
        e['reference_mode_folder']=str(trial/f'case_{tag}/court_seed_only')
        e['reference_initialization']=load(trial/f'case_{tag}/result.json')['modes']['court_seed_only']
        answer.append(e)
    for tag in ['016','017','018']:
        c=BASE/f'cohort_tracking/case_{tag}';t=BASE/f'court_initialization_transfer_{tag}'
        folder=c/f'tracking_quality_20261102_stage4/players_{tag}_temporal';s=load(folder/'summary.json')
        answer.append(dict(tag=tag,baseline_folder=str(folder),raw_csv=s['input_csv'],first=s['first_frame'],last=s['last_frame'],adaptive_size=True,cache=str(t),reference_mode_folder=str(t/'court_seed_only'),reference_initialization=load(t/'result.json')['modes']['court_seed_only'],role='known_AI_diagnosed_initial_nonplayer_failure' if tag=='018' else 'initialization_correspondence_control_not_certified_whole_tracking_accuracy'))
    return sorted(answer,key=lambda e:e['tag']),p

def cache(entry):
    folder=Path(entry['cache']);rows=read(folder/'cached_detections.csv');z=np.load(folder/'appearance_cache.npz');features=z['features'];offset=z['frame_offsets'];assert len(offset)==entry['last']-entry['first']+2
    raw=[r for r in read(Path(entry['raw_csv'])) if int(r['track_id'])>=0 and float(r['confidence'])>=.25]
    fields=['source_frame','track_id','x1','y1','x2','y2','confidence'];sig=lambda r:tuple(r[k] for k in fields)
    assert [sig(r) for r in rows]==[sig(r) for r in raw]
    return rows,features,offset

def main():
    OUT.mkdir();cases,previous=plans();cal=Path(previous['calibration']);h=np.array(load(cal)['image_to_court_homography'])
    protected={str(n):sha(n) for n in [cal,Path(__file__),ROOT/'court_initialization_guard.py',Path(old.a.__file__),Path(old.selector.__file__)]}
    for e in cases:
        for n in [Path(e['raw_csv']),Path(e['cache'])/'cached_detections.csv',Path(e['cache'])/'appearance_cache.npz',Path(e['baseline_folder'])/'events.json',Path(e['reference_mode_folder'])/'selected_four.csv']:protected[str(n)]=sha(n)
    dump(OUT/'diagnostic_protocol.json',dict(cases=cases,calibration=str(cal),source_video=previous['source_video'],source_video_sha256=previous['source_video_sha256'],selection='All18 retained cases: earlier15 plus016,017,018. At old court-filter initialization and preceding7 frames, enumerate every eligible near pair.',purpose='Hypothesis diagnosis before new rule outputs; same-match development. Clothing similarity is a cue, not identity GT. No outcomes, no individualID selection, no new human review.',independent_GT=False,protected_input_hashes=protected))
    pairs=[];candidates=[]
    for e in cases:
        rows,features,offset=cache(e);init=e['reference_initialization']['initialized_frame']
        for f in range(max(e['first'],init-7),init+1):
            i=f-e['first'];rs=rows[offset[i]:offset[i+1]];fs=features[offset[i]:offset[i+1]];near=[]
            for r,feat in zip(rs,fs):
                if not eligible(r,h) or float(r['bottom_center_y'])/2160<.61 or (float(r['y2'])-float(r['y1']))/2160<.08:continue
                xy=court_point(r,h);valid=not np.isnan(feat).all();near.append((r,feat if valid else None))
                candidates.append(dict(tag=e['tag'],source_frame=f,track_id=r['track_id'],confidence=r['confidence'],court_x=xy[0],court_y=xy[1],appearance_available=int(valid),reference_near_seed=int(int(r['track_id']) in [v for k,v in e['reference_initialization']['seed_ids'].items() if k.startswith('near_') ])))
            for (a,fa),(b,fb) in itertools.combinations(near,2):
                pairs.append(dict(tag=e['tag'],source_frame=f,at_reference_initialization=int(f==init),raw_id_a=a['track_id'],raw_id_b=b['track_id'],clothing_distance=old.a.app_distance(fa,fb),minimum_confidence=min(float(a['confidence']),float(b['confidence'])),mean_confidence=(float(a['confidence'])+float(b['confidence']))/2,overlap_fraction=old.overlap_fraction(a,b),both_features_available=int(fa is not None and fb is not None),reference_selected_pair=int({int(a['track_id']),int(b['track_id'])}=={v for k,v in e['reference_initialization']['seed_ids'].items() if k.startswith('near_')})))
    write(OUT/'initial_candidate_pairs.csv',pairs);write(OUT/'initial_near_candidates.csv',candidates)
    assert all(sha(Path(n))==value for n,value in protected.items())
    dump(OUT/'diagnostic_verification.json',dict(status='passed',cases=len(cases),pair_rows=len(pairs),candidate_rows=len(candidates),raw_cache_rows_exact=True,protected_sources_unchanged=True,identity_accuracy_established=False))
    for e in cases:
        options=[r for r in pairs if r['tag']==e['tag'] and r['at_reference_initialization']];options.sort(key=lambda r:r['clothing_distance'])
        print(json.dumps(dict(tag=e['tag'],pairs=[{k:r[k] for k in ['raw_id_a','raw_id_b','clothing_distance','mean_confidence','reference_selected_pair']} for r in options]),ensure_ascii=False))

if __name__=='__main__':main()
