"""Recalculate exported prefixes and summaries directly from protected observations."""
import csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'ball_quality_20261109/rally_player_prefix_expansion'
SLOTS=['near_front','near_rear','far_rear','far_front']

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def close(a,b):
    if a=='' or b=='':assert a==b,(a,b)
    else:assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(a,b)

def calculate_frame(cid,people,f,relative):
    row=dict(rally_candidate_id=cid,relative_source_frame=relative)
    for side in ['near','far']:
        p=[people[f][s] for s in SLOTS if s.startswith(side+'_') and s in people[f]]
        row[side+'_pair_valid']=int(len(p)==2)
        for name in ['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm']:row[side+'_'+name]=''
        if len(p)==2:
            xs=[float(r['bottom_center_x'])/3840 for r in p];ys=[float(r['bottom_center_y'])/2160 for r in p]
            row.update({side+'_center_x_norm':sum(xs)/2,side+'_center_y_norm':sum(ys)/2,
                side+'_x_gap_norm':max(xs)-min(xs),side+'_y_gap_norm':max(ys)-min(ys)})
    return row

def calculate_summary(rows):
    keys=[side+'_'+k for side in ['near','far'] for k in ['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm']]
    snap={k:rows[-1][k] for k in keys+['near_pair_valid','far_pair_valid']}
    summary=dict(snap)
    for key in keys:
        vals=[r[key] for r in rows if r[key]!='']
        summary[key+'_mean']=math.fsum(vals)/len(vals) if vals else ''
        summary[key+'_end_minus_start']=rows[-1][key]-rows[0][key] if rows[-1][key]!='' and rows[0][key]!='' else ''
    for side in ['near','far']:summary[side+'_pair_observed_fraction']=sum(r[side+'_pair_valid'] for r in rows)/len(rows)
    return snap,summary

def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for p,h in protocol['protected_input_hashes'].items():assert sha(Path(p))==h,p
    for n,h in result['artifact_hashes'].items():assert sha(OUT/n)==h,n
    assert sha(OUT/'executed_source.py')==protocol['script_sha256']
    meta=read(OUT/'rally_metadata.csv');outcomes={r['rally_candidate_id']:r for r in read(OUT/'human_outcomes.csv')}
    frames={(r['rally_candidate_id'],int(r['relative_source_frame'])):r for r in read(OUT/'prefix_frame_features.csv')}
    observations={(r['rally_candidate_id'],int(r['source_frame']),r['player_slot']):r for r in read(OUT/'requested_player_observations.csv')}
    snapshots={r['rally_candidate_id']:r for r in read(OUT/'snapshot_player_features.csv')}
    summaries={r['rally_candidate_id']:r for r in read(OUT/'prefix_player_summary_features.csv')}
    sensitivity={(r['rally_candidate_id'],r['feature']):r for r in read(OUT/'contact_timing_sensitivity.csv')}
    handoff=ROOT.parent/'frontcourt_handoff_20260927'
    latest={}
    for r in read(handoff/'human_review_log.csv'):
        if r['field']=='winner_camera_side':latest[r['candidate_id']]=r
    groups={r['candidate_id']:r for r in read(handoff/'candidate_point_group_map.csv')}
    assert groups['match1_rally_window_050']['point_group_id']==groups['match1_rally_window_051']['point_group_id']
    assert len(meta)==8 and len(frames)==248 and len(observations)==992
    assert Counter(r['winner_camera_side'] for r in outcomes.values())=={'near':5,'far':3}
    checked_values=0;variant_windows=0;available=0
    # Independent explicit checks protect against accidentally using the old048 bracket or invalid044 first serve.
    expected={'002':(2223,2229),'003':(4201,4223),'005':(8607,8618),'007':(15552,15563),
        '025':(round(1000.601100*60000/1001),)*2,'044':(115365,115373),'048':(125859,125881),'056':(154081,154086)}
    for m in meta:
        cid=m['rally_candidate_id'];tag=cid[-3:];a,b=expected[tag]
        assert (int(m['anchor_first_source_frame']),int(m['anchor_last_source_frame']))==(a,b)
        assert int(m['input_first_source_frame'])==b+30 and int(m['input_end_source_frame'])==b+90
        assert m['point_group_id']==groups[cid]['point_group_id']==outcomes[cid]['point_group_id']
        assert outcomes[cid]['winner_camera_side']==latest[cid]['value'] and outcomes[cid]['human_review_id']==latest[cid]['review_id']
        people=defaultdict(dict)
        for part in json.loads(m['player_source_parts_json']):
            for p in read(Path(part['path'])):
                if p['player_slot'] in part['slots']:
                    f=int(p['source_frame']);assert p['player_slot'] not in people[f];people[f][p['player_slot']]=p
        t0=b+30;rows=[]
        for offset in range(0,61,2):
            f=t0+offset;pr=people[f];row=calculate_frame(cid,people,f,offset);rows.append(row)
            actual=frames[cid,offset]
            assert actual['rally_candidate_id']==cid
            for k,v in row.items():
                if k!='rally_candidate_id':close(actual[k],v);checked_values+=1
            assert len({p['track_id'] for p in pr.values()})==len(pr)
            for slot in SLOTS:
                ob=observations[cid,f,slot];p=pr.get(slot);assert int(ob['observation_present'])==int(p is not None)
                for k in ['bottom_center_x','bottom_center_y','track_id','position_source','identity_status']:
                    assert ob[k]==(p.get(k,'') if p else ''),(cid,f,slot,k)
                available+=int(p is not None)
        snap,summ=calculate_summary(rows)
        for actual,calculated in [(snapshots[cid],snap),(summaries[cid],summ)]:
            assert set(actual)==set(calculated)|{'rally_candidate_id'}
            for k,v in calculated.items():close(actual[k],v);checked_values+=1
        assert int(m['four_observed_times'])==sum(len(people[t0+d])==4 for d in range(0,61,2))
        for side in ['near','far']:assert int(m[side+'_pair_observed_times'])==sum(r[side+'_pair_valid'] for r in rows)
        assert int(m['times_before_export_start'])==sum(t0+d<int(m['export_first_frame']) for d in range(0,61,2))
        if m['inherited_end_bracket_available']=='1':assert b+90<int(m['earliest_point_end_candidate'])
        variants=[]
        for contact in range(a,b+1):
            variant_windows+=1
            _,v=calculate_summary([calculate_frame(cid,people,contact+30+d,d) for d in range(0,61,2)]);variants.append(v)
        for k in summ:
            vals=[v[k] for v in variants if v[k]!=''];actual=sensitivity[cid,k]
            assert int(actual['bracket_variant_count'])==len(variants) and int(actual['available_variant_count'])==len(vals)
            for key,val in [('value_min',min(vals) if vals else ''),('value_max',max(vals) if vals else ''),('range_norm',max(vals)-min(vals) if vals else '')]:close(actual[key],val);checked_values+=1
    schema=json.loads((OUT/'feature_schema.json').read_text(encoding='utf-8'))
    assert set(schema['snapshot_features'])==set(next(iter(snapshots.values())))-{'rally_candidate_id'}
    assert set(schema['summary_features'])==set(next(iter(summaries.values())))-{'rally_candidate_id'}
    assert all(not any(x in k for x in ['winner','source_frame','time_sec','file','group','anchor','end_bracket']) for k in schema['summary_features'])
    assert protocol['no_new_human_annotation'] and not protocol['canonical_tracking_changed']
    report=dict(status='passed',recalculated_feature_values=checked_values,recalculated_contact_variant_windows=variant_windows,
        requested_player_slots=len(observations),exported_observations_present=available,missing_slots=len(observations)-available,
        all_source_and_artifact_hashes_match=True,latest_human_side_labels_match=True,corrected048_anchor_and025_outcome_preserved=True,
        same_point050_051_group_preserved=True,feature_allowlist_checked=True,
        limitation='Data integrity and deterministic feature construction verified; not independent identity/position accuracy or pre-outcome confirmation of all eight candidates.')
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
