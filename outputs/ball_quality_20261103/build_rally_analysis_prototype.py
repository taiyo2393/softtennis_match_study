"""Join reviewed ball data,player candidates,and human outcomes;no two-point fitting."""
import itertools,json,math,shutil
from collections import Counter,defaultdict
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha,cv2,np

BASE=ROOT.parent/'ball_quality_20261109'
OUT=BASE/'rally_analysis_prototype'
SLOTS=('near_front','near_rear','far_rear','far_front')
NAMES=dict(zip(SLOTS,('N1','N2','F1','F2')))
PAIR_FEATURES=[side+'_'+name for side in ['near','far'] for name in ['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm']]
CONFIGS=[dict(tag='044',ball_folder=BASE/'rally_repair_pilot_044_v2/completed_review',
    player_path=ROOT/'person_044_pose_recovery/user_reviewed/observed_player_paths.csv',core_start=115374,end_first=115776,
    identity_evidence='Far-pair video correspondence and11recovered boxes confirmed;not independent full-frame position GT.'),
    dict(tag='056',ball_folder=BASE/'rally_repair_056/completed_review',
    player_path=ROOT.parent/'tracking_quality_20261102/players_056_appearance_motion_guarded/selected_four.csv',core_start=154087,end_first=154840,
    identity_evidence='Candidate appearance/motion path;prior seed/junction checks,not all-clip independent identity GT.')]


def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')


def pair_features(players):
    result={}
    for side in ['near','far']:
        points=[(float(players[s]['bottom_center_x'])/3840,float(players[s]['bottom_center_y'])/2160) for s in SLOTS if s.startswith(side+'_') and s in players]
        result[side+'_pair_valid']=int(len(points)==2)
        for name in ['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm']:result[side+'_'+name]=''
        if len(points)==2:
            a,b=points
            result.update({side+'_center_x_norm':(a[0]+b[0])/2,side+'_center_y_norm':(a[1]+b[1])/2,
                side+'_x_gap_norm':abs(a[0]-b[0]),side+'_y_gap_norm':abs(a[1]-b[1])})
    return result


def source_players(path):
    grouped=defaultdict(dict)
    for r in read(path):
        slot=r['player_slot'];assert slot in SLOTS
        assert 'interpolat' not in r.get('position_source','').lower()
        f=int(r['source_frame']);assert slot not in grouped[f]
        assert all(math.isfinite(float(r[k])) for k in ['bottom_center_x','bottom_center_y','x1','y1','x2','y2'])
        grouped[f][slot]=r
    return grouped


def frame_feature(tag,f,players,ball):
    result=dict(rally_candidate_id=f'match1_rally_window_{tag}',source_frame=f,**pair_features(players),all_four_observed=int(len(players)==4),
        ball_observation_present=int(ball['observation_present']),ball_x_norm='',ball_y_norm='')
    if int(ball['observation_present']):result.update(ball_x_norm=float(ball['x_px'])/1920,ball_y_norm=float(ball['y_px'])/1080)
    return result


def prefix_features(tag,times,players,balls):
    return [dict(frame_feature(tag,f,players.get(f,{}),balls[f]),relative_source_frame=f-times[0]) for f in times]


def summarize_prefix(tag,rows):
    snapshot=dict(rally_candidate_id=f'match1_rally_window_{tag}',**{k:rows[-1][k] for k in PAIR_FEATURES+['near_pair_valid','far_pair_valid']})
    summary=dict(snapshot)
    for key in PAIR_FEATURES:
        available=[float(r[key]) for r in rows if r[key]!='']
        summary[key+'_mean']=float(np.mean(available)) if available else ''
        summary[key+'_end_minus_start']=float(rows[-1][key])-float(rows[0][key]) if rows[0][key]!='' and rows[-1][key]!='' else ''
    for side in ['near','far']:summary[side+'_pair_observed_fraction']=sum(r[side+'_pair_valid'] for r in rows)/len(rows)
    with_ball=dict(summary)
    for key in ['ball_x_norm','ball_y_norm']:
        available=[float(r[key]) for r in rows if r[key]!='']
        with_ball[key+'_last_requested_time']=rows[-1][key]
        with_ball[key+'_mean']=float(np.mean(available)) if available else ''
    with_ball['ball_observed_fraction']=sum(r['ball_observation_present'] for r in rows)/len(rows)
    with_ball['ball_last_requested_time_observed']=rows[-1]['ball_observation_present']
    return snapshot,summary,with_ball


def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve this feature construction and source snapshot.'
    label_path=ROOT.parent/'frontcourt_handoff_20260927/rally_label_review.csv'
    log_path=label_path.parent/'human_review_log.csv';group_path=label_path.parent/'candidate_point_group_map.csv'
    labels=read(label_path);groups={r['candidate_id']:r for r in read(group_path)}
    human=[r for r in labels if r['winner_camera_side_review_status']=='human_confirmed']
    by_label={r['candidate_id']:r for r in human};logs=read(log_path)
    latest={}
    for r in logs:
        if r['field']=='winner_camera_side':latest[r['candidate_id']]=r
    assert len(human)==11 and Counter(r['winner_camera_side'] for r in human)=={'near':7,'far':4}
    assert all(latest[r['candidate_id']]['value']==r['winner_camera_side'] for r in human)
    paths=[label_path,log_path,group_path,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        BASE/'two_rally_repair_assessment/assessment.json']
    for c in CONFIGS:paths+=[c['player_path']]+[c['ball_folder']/n for n in ['result.json','verification.json','assisted_states_sampled.csv','cases.json','review_labels.csv']]
    protected={str(p):sha(p) for p in paths}
    protocol=dict(status='fixed_before_new_join_or_feature_extraction',date_jst='2026-11-09',
        purpose='Two-point construction prototype plus inventory of11existing human side labels;not fitted outcome model or detector benchmark.',
        primary_question='Can an early fixed-duration doubles arrangement provide score-side information,and does observable ball data add value? Current two same-class cases cannot answer.',
        input_time='For each reviewed candidate interior,t0=first sampled ball frame(core_start+1);31observations at stride2,t0..t0+60(~1.001s). Snapshot is last REQUESTED time,not best/latest available frame. End strictly before inherited earliest possible point-end candidate.',
        pair_representation='Per side,centroid x/y and absolute x/y gaps of two observed bbox-bottom proxies;invariant to within-pair slot swaps. Missing teammate means that pair features stay missing. Does not prevent assignment to wrong person or wrong court side.',
        coordinates='Original player boxes3840x2160=>1920x1080for joining/render;normalize x/3840,y/2160for features. Ball1920x1080=>normalizedimagexy. No homography,metres,true feet,3Dball position or physical speed claim.',
        comparisons=['majority-side baseline fitted on training groups only','last requested time player arrangement','fixed1second player arrangement summary','same summary plus ball coordinates/observation masks'],
        feature_isolation='Winner/end brackets/total rally duration/repair decisions/time/whole-rally availability/sourceframeIDs and filenames are NEVER model feature columns. Outcomes stored separately. All approved feature values derive only from requested early prefix.',
        future_processing='Source person tracking may be bidirectional;ball median and human review use whole clips. Pre-score coordinates are selected,but end-to-end causal or real-time prediction is NOT established.',
        evaluation_plan='Define a larger rally dataset before fitting. Keep point groups intact(050/051samepoint),split by match when available;within-match blocked/grouped exploration must be labelled as such. Fit normalization/imputation/model selection on training groups only. No random frame split or adjacent-frame independent trials.',
        sufficiency='Current joined examples044/056both near-winner;no classification,association,p-value or tactical-cause conclusion. Reuse11human labels,not eleven guaranteed independent complete rallies.',
        no_new_review_requests=True,new_model_fit=False,canonical_tracking_changed=False,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol);shutil.copy2(Path(__file__),OUT/'executed_source.py')
    # Existing continuous exports are inventoried,not silently declared independently validated.
    catalog={}
    for tag in ['002','003','005','007','048']:
        p=ROOT.parent/f'tracking_quality_20261102_stage9/players_{tag}_temporal/four_player_path.csv'
        assert p.exists();catalog[tag]=[dict(path=str(p),slots=list(SLOTS))]
    base10=ROOT.parent/'tracking_quality_20261102_stage10/experiment'
    catalog['025']=[dict(path=str(base10/'tracking_quality_20261102_stage4/players_025_temporal/selected_four.csv'),slots=list(SLOTS[:2])),
        dict(path=str(base10/'tracking_quality_20261102_stage7/players_025_global/far_pair_path.csv'),slots=list(SLOTS[2:]))]
    for c in CONFIGS:catalog[c['tag']]=[dict(path=str(c['player_path']),slots=list(SLOTS))]
    catalog_hashes={}
    for parts in catalog.values():
        for part in parts:
            p=Path(part['path']);assert p.exists();catalog_hashes[str(p)]=sha(p)
    inventory=[]
    for r in labels:
        cid=r['candidate_id'];tag=cid[-3:];confirmed=cid in by_label
        inventory.append(dict(rally_candidate_id=cid,point_group_id=groups[cid]['point_group_id'],point_group_status=groups[cid]['group_status'],
            human_winner_side_available=int(confirmed),winner_camera_side=r['winner_camera_side'] if confirmed else '',
            latest_human_side_review_id=latest[cid]['review_id'] if confirmed else '',
            serve_review_status=r['serve_review_status'],boundary_review_status=r['boundary_review_status'],
            continuous_player_export_found=int(tag in catalog),player_source_parts_json=json.dumps(catalog.get(tag,[])),
            assisted_ball_snapshot_found=int(tag in {'044','056'}),current_joined_prototype=int(tag in {'044','056'}),
            stage='joined_construction_example' if tag in {'044','056'} else 'human_outcome_and_player_export' if confirmed and tag in catalog else 'human_outcome_only_or_snapshot' if confirmed else 'outcome_pending',
            note='Missing export/pending fields are not proof of invisible players or invalid play. No source analysis_eligible field overwritten.'))
    write(OUT/'existing_rally_inventory.csv',inventory);dump(OUT/'player_export_catalog_hashes.json',catalog_hashes)
    outcome_rows=[]
    for r in human:
        cid=r['candidate_id'];outcome_rows.append(dict(rally_candidate_id=cid,point_group_id=groups[cid]['point_group_id'],winner_camera_side=r['winner_camera_side'],
            latest_human_review_id=latest[cid]['review_id'],winner_team_derived_not_independent_label=r['winner_team_id'],source_file=str(label_path),source_sha256=sha(label_path)))
    write(OUT/'human_outcomes.csv',outcome_rows)
    all_joined=[];all_features=[];snapshots=[];summaries=[];ball_summaries=[];rally_meta=[];poster_cells=[]
    invariance_tests=0;feature_isolation_tests=0
    for config in CONFIGS:
        tag=config['tag'];folder=config['ball_folder'];ball_result=load(folder/'result.json')
        assert load(folder/'verification.json')['status']=='passed'
        assert all(sha(folder/n)==h for n,h in ball_result['export_hashes'].items())
        balls={int(r['source_frame']):r for r in read(folder/'assisted_states_sampled.csv')}
        cases={int(r['source_frame']):r for r in load(folder/'cases.json')}
        people=source_players(config['player_path']);first=min(balls);times=list(range(first,first+61,2));assert len(times)==31 and first==config['core_start']+1
        assert set(times)<=set(balls) and times[-1]<config['end_first']
        for f,b in sorted(balls.items()):
            ps=people.get(f,{});row=dict(rally_candidate_id=f'match1_rally_window_{tag}',source_frame=f,
                all_four_observed=int(len(ps)==4),player_observed_count=len(ps),ball_observation_present=int(b['observation_present']),ball_x_px=b['x_px'],ball_y_px=b['y_px'],
                ball_coordinate_source=b['coordinate_source'],ball_review_decision=b['decision'],ball_independent_new_center_gt=0)
            for slot in SLOTS:
                name=NAMES[slot];p=ps.get(slot)
                for k in ['observed','x_px','y_px','track_id','position_source','identity_status','human_review_id']:row[name+'_'+k]=''
                row[name+'_observed']=int(p is not None)
                if p:row.update({name+'_x_px':float(p['bottom_center_x'])*.5,name+'_y_px':float(p['bottom_center_y'])*.5,
                    name+'_track_id':p['track_id'],name+'_position_source':p['position_source'],name+'_identity_status':p.get('identity_status',''),name+'_human_review_id':p.get('human_video_review_id','')})
            all_joined.append(row)
        features=prefix_features(tag,times,people,balls)
        for f in times:
            ps=people.get(f,{})
            for swap_near,swap_far in itertools.product([False,True],repeat=2):
                swapped=dict(ps)
                for do_swap,a,b in [(swap_near,'near_front','near_rear'),(swap_far,'far_front','far_rear')]:
                    if do_swap:
                        va,vb=ps.get(a),ps.get(b);swapped.pop(a,None);swapped.pop(b,None)
                        if va is not None:swapped[b]=va
                        if vb is not None:swapped[a]=vb
                assert pair_features(ps)==pair_features(swapped);invariance_tests+=1
        # Future observations and the opposite outcome must not affect approved prefix inputs.
        extra_people=dict(people);extra_people[times[-1]+2]={s:dict(bottom_center_x='1',bottom_center_y='1') for s in SLOTS}
        extra_balls=dict(balls);extra_balls[times[-1]+2]=dict(observation_present='1',x_px='1',y_px='1')
        assert prefix_features(tag,times,extra_people,extra_balls)==features;feature_isolation_tests+=1
        snap,summary,with_ball=summarize_prefix(tag,features);snapshots.append(snap);summaries.append(summary);ball_summaries.append(with_ball)
        all_features+=features;label=by_label[f'match1_rally_window_{tag}']
        rally_meta.append(dict(rally_candidate_id=label['candidate_id'],winner_camera_side=label['winner_camera_side'],
            point_group_id=groups[label['candidate_id']]['point_group_id'],input_first_source_frame=times[0],input_end_source_frame=times[-1],
            requested_times=31,prefix_span_seconds=60/(60000/1001),earliest_point_end_candidate=config['end_first'],end_candidate_margin_frames=config['end_first']-times[-1],
            prefix_frames_with_four_observations=sum(r['all_four_observed'] for r in features),prefix_ball_observation_times=sum(r['ball_observation_present'] for r in features),
            whole_sample_times=len(balls),whole_sample_with_four_observations=sum(len(people.get(f,{}))==4 for f in balls),
            identity_evidence=config['identity_evidence'],independent_position_accuracy_established=0,
            player_source=str(config['player_path']),player_source_sha256=sha(config['player_path']),ball_source=str(folder/'assisted_states_sampled.csv'),ball_source_sha256=sha(folder/'assisted_states_sampled.csv')))
        for index in [0,15,30]:
            f=times[index];c=cases[f];assert sha(Path(c['source_image']))==c['source_image_sha256']
            image=cv2.imread(c['source_image']);assert image.shape==(1080,1920,3)
            tile=np.full((410,640,3),22,np.uint8);tile[50:]=cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA)
            ps=people.get(f,{})
            for slot,p in ps.items():
                color=(0,170,255) if slot.startswith('near') else (255,180,50)
                x,y=round(float(p['bottom_center_x'])/6),50+round(float(p['bottom_center_y'])/6)
                cv2.circle(tile,(x,y),5,color,2,cv2.LINE_AA)
            b=balls[f]
            if int(b['observation_present']):cv2.circle(tile,(round(float(b['x_px'])/3),50+round(float(b['y_px'])/3)),5,(70,255,90),2,cv2.LINE_AA)
            cv2.putText(tile,f'{tag} | t={index*2/(60000/1001):.3f}s | source {f}',(10,21),0,.54,(240,240,240),1,cv2.LINE_AA)
            cv2.putText(tile,f'Players {len(ps)}/4 | ball {int(b["observation_present"])} | image proxies',(10,42),0,.50,(220,220,220),1,cv2.LINE_AA);poster_cells.append(tile)
    write(OUT/'joined_sampled_observations.csv',all_joined);write(OUT/'prefix_frame_features.csv',all_features)
    write(OUT/'snapshot_player_features.csv',snapshots);write(OUT/'prefix_player_summary_features.csv',summaries);write(OUT/'prefix_player_ball_summary_features.csv',ball_summaries)
    write(OUT/'prototype_rally_metadata.csv',rally_meta)
    assert cv2.imwrite(str(OUT/'prefix_diagnostic.jpg'),np.vstack([np.hstack(poster_cells[:3]),np.hstack(poster_cells[3:])]))
    schema={}
    forbidden={'winner_camera_side','winner_team_id','source_frame','relative_source_frame','rally_candidate_id','point_group_id','earliest_point_end_candidate','whole_sample_times','ball_review_decision'}
    for arm,rows in [('snapshot_player',snapshots),('prefix_player_summary',summaries),('prefix_player_ball_summary',ball_summaries)]:
        keys=[k for k in rows[0] if k!='rally_candidate_id'];assert not set(keys)&forbidden
        schema[arm]=dict(model_input_columns=keys,join_key_not_input='rally_candidate_id',target_source='human_outcomes.csv',missing_value='empty string;imputation only within future training folds')
    dump(OUT/'feature_schema.json',schema)
    # Explicit target-flip test is possible because the feature files contain no target columns.
    assert all('winner_camera_side' not in r for r in all_features+snapshots+summaries+ball_summaries)
    assert all(sha(Path(p))==h for p,h in protected.items())
    assert all(sha(Path(p))==h for p,h in catalog_hashes.items())
    result=dict(status='joined_rally_feature_construction_complete',joined_rallies=2,joined_sampled_times=len(all_joined),prefix_observations=len(all_features),
        human_side_label_candidates=len(human),human_side_label_counts=dict(Counter(r['winner_camera_side'] for r in human)),
        human_side_labels_with_continuous_player_export=sum(r['human_winner_side_available'] and r['continuous_player_export_found'] for r in inventory),
        prototype_outcome_counts=dict(Counter(r['winner_camera_side'] for r in rally_meta)),rally_metadata=rally_meta,
        paired_slot_permutation_checks=invariance_tests,future_observation_injection_checks=feature_isolation_tests,
        feature_schema_excludes_outcomes_and_future_metadata=True,coordinate_processing_is_offline=True,
        model_fitting_performed=False,outcome_association_estimated=False,causal_loss_explanation_established=False,
        no_new_human_annotation=True,canonical_tracking_changed=False,
        conclusion='Reusable data structure and within-pair-ID-invariant features demonstrated. Two same-class cases cannot test outcome discrimination;existing11labels and8player exports are inventory,not eleven fully qualified independent rallies.',
        artifact_hashes={n:sha(OUT/n) for n in ['existing_rally_inventory.csv','human_outcomes.csv','joined_sampled_observations.csv','prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv','prefix_player_ball_summary_features.csv','prototype_rally_metadata.csv','feature_schema.json','prefix_diagnostic.jpg']})
    dump(OUT/'result.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
