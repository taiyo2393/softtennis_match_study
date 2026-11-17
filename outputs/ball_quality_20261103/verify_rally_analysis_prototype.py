"""Recheck exact joins,source-side labels,time cutoffs,and image-coordinate features."""
import json,math
from collections import Counter,defaultdict
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,load,sha,np

OUT=ROOT.parent/'ball_quality_20261109/rally_analysis_prototype'
MAPPING={'N1':'near_front','N2':'near_rear','F1':'far_rear','F2':'far_front'}
KEYS=[side+'_'+name for side in ['near','far'] for name in ['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm']]


def close(a,b):
    if a=='' or b=='':assert a==b
    else:assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(a,b)


def main():
    p,r=load(OUT/'protocol.json'),load(OUT/'result.json')
    assert sha(OUT/'executed_source.py')==p['script_sha256']
    assert all(sha(Path(path))==h for path,h in p['protected_input_hashes'].items())
    assert all(sha(OUT/name)==h for name,h in r['artifact_hashes'].items())
    catalog=load(OUT/'player_export_catalog_hashes.json');assert all(sha(Path(path))==h for path,h in catalog.items())
    observations=read(OUT/'joined_sampled_observations.csv');features=read(OUT/'prefix_frame_features.csv');meta=read(OUT/'prototype_rally_metadata.csv')
    assert len(observations)==577 and len(features)==62 and len(meta)==2
    labels=read(OUT/'human_outcomes.csv');assert len(labels)==11 and Counter(x['winner_camera_side'] for x in labels)=={'near':7,'far':4}
    source_labels={x['candidate_id']:x for x in read(ROOT.parent/'frontcourt_handoff_20260927/rally_label_review.csv')}
    log=read(ROOT.parent/'frontcourt_handoff_20260927/human_review_log.csv');latest={}
    for x in log:
        if x['field']=='winner_camera_side':latest[x['candidate_id']]=x
    for x in labels:
        assert source_labels[x['rally_candidate_id']]['winner_camera_side_review_status']=='human_confirmed'
        assert x['winner_camera_side']==latest[x['rally_candidate_id']]['value']
        assert x['latest_human_review_id']==latest[x['rally_candidate_id']]['review_id']
    join_checks=0;feature_checks=0
    for m in meta:
        rid=m['rally_candidate_id'];sources=defaultdict(dict)
        for q in read(Path(m['player_source'])):
            key=(int(q['source_frame']),q['player_slot']);assert q['player_slot'] not in sources[key[0]]
            sources[key[0]][q['player_slot']]=q
        balls={int(x['source_frame']):x for x in read(Path(m['ball_source']))}
        byframe={int(x['source_frame']):x for x in observations if x['rally_candidate_id']==rid}
        assert set(byframe)==set(balls)
        for f,x in byframe.items():
            q=sources.get(f,{})
            assert int(x['player_observed_count'])==len(q) and int(x['all_four_observed'])==int(len(q)==4)
            assert int(x['ball_observation_present'])==int(balls[f]['observation_present'])
            close(x['ball_x_px'],balls[f]['x_px']);close(x['ball_y_px'],balls[f]['y_px'])
            for name,slot in MAPPING.items():
                assert int(x[name+'_observed'])==int(slot in q)
                if slot in q:
                    close(x[name+'_x_px'],float(q[slot]['bottom_center_x'])/2);close(x[name+'_y_px'],float(q[slot]['bottom_center_y'])/2)
                    assert x[name+'_track_id']==q[slot]['track_id'] and x[name+'_position_source']==q[slot]['position_source']
                else:assert x[name+'_x_px']==x[name+'_y_px']==''
            join_checks+=1
        prefix=sorted([x for x in features if x['rally_candidate_id']==rid],key=lambda x:int(x['source_frame']))
        requested=list(range(int(m['input_first_source_frame']),int(m['input_end_source_frame'])+1,2))
        assert len(prefix)==len(requested)==31 and [int(x['source_frame']) for x in prefix]==requested
        assert requested[-1]-requested[0]==60 and requested[-1]<int(m['earliest_point_end_candidate'])
        for x in prefix:
            f=int(x['source_frame']);joined=byframe[f]
            assert int(x['relative_source_frame'])==f-requested[0]
            for side,names in [('near',['N1','N2']),('far',['F1','F2'])]:
                valid=all(int(joined[n+'_observed']) for n in names);assert int(x[side+'_pair_valid'])==int(valid)
                if valid:
                    a,b=[(float(joined[n+'_x_px'])/1920,float(joined[n+'_y_px'])/1080) for n in names]
                    expected=[(a[0]+b[0])/2,(a[1]+b[1])/2,abs(a[0]-b[0]),abs(a[1]-b[1])]
                    for key,value in zip(['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm'],expected,strict=True):close(x[side+'_'+key],value)
                else:assert all(x[side+'_'+key]=='' for key in ['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm'])
            assert x['ball_observation_present']==joined['ball_observation_present']
            close(x['ball_x_norm'],float(joined['ball_x_px'])/1920 if joined['ball_x_px']!='' else '')
            close(x['ball_y_norm'],float(joined['ball_y_px'])/1080 if joined['ball_y_px']!='' else '')
            feature_checks+=1
        snapshot=next(x for x in read(OUT/'snapshot_player_features.csv') if x['rally_candidate_id']==rid)
        summary=next(x for x in read(OUT/'prefix_player_summary_features.csv') if x['rally_candidate_id']==rid)
        with_ball=next(x for x in read(OUT/'prefix_player_ball_summary_features.csv') if x['rally_candidate_id']==rid)
        for key in KEYS:
            close(snapshot[key],prefix[-1][key]);close(summary[key],snapshot[key]);close(with_ball[key],snapshot[key])
            vals=[float(x[key]) for x in prefix if x[key]!='']
            mean=sum(vals)/len(vals) if vals else ''
            delta=float(prefix[-1][key])-float(prefix[0][key]) if prefix[0][key]!='' and prefix[-1][key]!='' else ''
            close(summary[key+'_mean'],mean);close(summary[key+'_end_minus_start'],delta)
            close(with_ball[key+'_mean'],mean);close(with_ball[key+'_end_minus_start'],delta)
        for key in ['ball_x_norm','ball_y_norm']:
            vals=[float(x[key]) for x in prefix if x[key]!='']
            close(with_ball[key+'_mean'],sum(vals)/len(vals) if vals else '');close(with_ball[key+'_last_requested_time'],prefix[-1][key])
        close(with_ball['ball_observed_fraction'],sum(int(x['ball_observation_present']) for x in prefix)/31)
        assert sum(int(x['all_four_observed']) for x in prefix)==int(m['prefix_frames_with_four_observations'])
        assert sum(int(x['ball_observation_present']) for x in prefix)==int(m['prefix_ball_observation_times'])
    schema=load(OUT/'feature_schema.json')
    forbidden=['winner','source_frame','point_group','end_candidate','whole_sample','repair','decision','source_file','duration','confidence']
    for arm,s in schema.items():
        assert not any(any(word in key for word in forbidden) for key in s['model_input_columns'])
    assert r['prototype_outcome_counts']=={'near':2} and not r['model_fitting_performed'] and not r['outcome_association_estimated']
    inventory=read(OUT/'existing_rally_inventory.csv');assert len(inventory)==63
    assert sum(int(x['human_winner_side_available']) for x in inventory)==11
    assert sum(int(x['human_winner_side_available']) and int(x['continuous_player_export_found']) for x in inventory)==8
    groups={x['rally_candidate_id']:x['point_group_id'] for x in inventory}
    assert groups['match1_rally_window_050']==groups['match1_rally_window_051']
    assert all(sha(Path(path))==h for path,h in p['protected_input_hashes'].items())
    result=dict(status='passed',joined_frame_rows_recomputed=join_checks,prefix_feature_rows_recomputed=feature_checks,
        snapshot_and_summary_values_recomputed=True,source_side_labels_and_latest_corrections_verified=11,
        prefix_cutoffs_before_end_candidates=True,missing_pairs_and_ball_states_preserved=True,
        approved_input_schema_excludes_outcome_and_future_metadata=True,same_point_group_050_051_preserved=True,
        source_and_artifact_hashes_unchanged=True,model_fitting_performed=False,independent_accuracy_not_claimed=True,script_sha256=sha(Path(__file__)))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
