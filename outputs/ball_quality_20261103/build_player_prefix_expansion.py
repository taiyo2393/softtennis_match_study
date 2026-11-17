"""Construct an eight-candidate offline input dataset, without fitting outcomes."""
import csv, hashlib, itertools, json, math, shutil
from collections import Counter, defaultdict
from pathlib import Path
from build_rally_analysis_prototype import pair_features, summarize_prefix, SLOTS, PAIR_FEATURES

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'rally_player_prefix_expansion'
HANDOFF = ROOT.parent / 'frontcourt_handoff_20260927'
FPS = 60000 / 1001

def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def write(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def players_from_parts(parts):
    people = defaultdict(dict)
    for part in parts:
        for r in read(Path(part['path'])):
            slot = r['player_slot']
            if slot not in part['slots']:
                continue
            frame = int(r['source_frame'])
            assert slot not in people[frame], (frame, slot)
            assert 'interpolat' not in r.get('position_source', '').lower()
            assert all(math.isfinite(float(r[k])) for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y'])
            assert abs(float(r['timestamp_sec']) - frame / FPS) < 1e-5
            people[frame][slot] = r
    for frame, ps in people.items():
        assert len({r['track_id'] for r in ps.values()}) == len(ps), frame
    return people

def features(cid, start, people):
    return [dict(rally_candidate_id=cid, relative_source_frame=offset,
                 **pair_features(people.get(start+offset, {}))) for offset in range(0,61,2)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Preserve existing snapshot.'
    inventory = read(BASE/'rally_analysis_prototype/existing_rally_inventory.csv')
    selected = [r for r in inventory if r['continuous_player_export_found']=='1' and r['human_winner_side_available']=='1']
    assert len(selected)==8
    labels = {r['candidate_id']:r for r in read(HANDOFF/'rally_label_review.csv')}
    latest = {}
    for r in read(HANDOFF/'human_review_log.csv'):
        if r['field']=='winner_camera_side': latest[r['candidate_id']]=r
    brackets = {r['candidate_id']:r for r in read(HANDOFF/'six_service_contact_brackets.csv')}
    revised048 = read(HANDOFF/'048_selected_position_candidates.csv')
    p044 = json.loads((ROOT/'play_start_audit/044_scope_report.json').read_text(encoding='utf-8'))
    p056 = json.loads((BASE/'rally_repair_056/protocol.json').read_text(encoding='utf-8'))
    provenance_paths = [BASE/'rally_analysis_prototype/existing_rally_inventory.csv', HANDOFF/'rally_label_review.csv',
        HANDOFF/'human_review_log.csv', HANDOFF/'candidate_point_group_map.csv', HANDOFF/'six_service_contact_brackets.csv',
        HANDOFF/'048_selected_position_candidates.csv', HANDOFF/'detect_additional_labeled_snapshots.py',
        ROOT/'play_start_audit/044_scope_report.json', BASE/'rally_repair_056/protocol.json']
    canonical_paths = [ROOT/n for n in ['reviewed_ball_overrides.json','reviewed_person_overrides.json','reviewed_tracking_bundles.json']]
    for r in selected:
        provenance_paths += [Path(p['path']) for p in json.loads(r['player_source_parts_json'])]
    anchors = {}
    for r in selected:
        cid=r['rally_candidate_id']; tag=cid[-3:]
        if tag in {'002','003','005','007'}:
            a=brackets[cid]
            anchors[tag]=dict(first=int(a['contact_candidate_first_frame']),last=int(a['contact_candidate_last_frame']),
                stratum='ai_service_sequence',source=str(HANDOFF/'six_service_contact_brackets.csv'),
                evidence=a['evidence'],validity='AI sequence supported; human validity pending',end_first=None)
        elif tag=='048':
            a=revised048[0]
            assert all((x['contact_first_possible_frame'],x['contact_last_possible_frame'])==(a['contact_first_possible_frame'],a['contact_last_possible_frame']) for x in revised048)
            anchors[tag]=dict(first=int(a['contact_first_possible_frame']),last=int(a['contact_last_possible_frame']),
                stratum='ai_service_sequence',source=str(HANDOFF/'048_selected_position_candidates.csv'),
                evidence=a['evidence'],validity='AI revised service bracket; human validity pending',end_first=None)
        elif tag=='025':
            a=labels[cid]; assert a['serve_review_status']=='impact_confirmed'
            f=round(float(a['reviewed_serve_impact_sec'])*FPS)
            anchors[tag]=dict(first=f,last=f,stratum='legacy_manual_contact',source=str(HANDOFF/'rally_label_review.csv'),
                evidence='reviewed_serve_impact_sec; legacy_start_method=manual_valid_serve_impact; historical snapshot script uses round(sec*60000/1001)',
                validity='Legacy manual impact record reused; effective validity and precision not independently reverified',end_first=None)
        elif tag=='044':
            first,last=p044['valid_serve_contact_source_frame_interval']
            anchors[tag]=dict(first=first,last=last,stratum='human_valid_service_ai_timing',source=str(ROOT/'play_start_audit/044_scope_report.json'),
                evidence='Human valid second serve; AI contact/end brackets',validity='Human confirmed valid second serve; exact contact pending',
                end_first=p044['point_end_source_frame_interval'][0])
        elif tag=='056':
            first,last=p056['interval']['contact_bracket']
            anchors[tag]=dict(first=first,last=last,stratum='human_valid_service_ai_timing',source=str(BASE/'rally_repair_056/protocol.json'),
                evidence='Human valid opening serve; inherited broad contact/end brackets retained',validity='Human confirmed valid opening serve; exact contact pending',
                end_first=p056['interval']['end_bracket'][0])
        else: raise AssertionError(tag)
    protected={str(p):sha(p) for p in set(provenance_paths+canonical_paths)}
    dump(OUT/'protocol.json',dict(status='fixed_before_feature_extraction',date_jst='2026-11-09',anchors=anchors,
        selection='All eight inventoried candidates with existing human winner-side labels and continuous player exports. Convenience sample of one match; not outcome-blind or population representative.',
        sampling='t0=last inherited possible contact frame+30;31 requested times at stride2 through t0+60. Never move to first/best tracked frame. Legacy025 nominal rounded timestamp has no established uncertainty interval.',
        relation_to_previous_prototype='New common 0.5-to-1.5-second post-contact window, not the previous044/056 immediate post-contact window. Older outputs remain unchanged.',
        timing='~0.5005sec after last possible contact to ~1.5015sec after it. Actual contact-relative timing varies with bracket width. All contact brackets remain provisional.',
        representations='Eight normalized image-coordinate pair centroid/absolute-gap features; bbox-bottom proxies, not metres/true feet. Missing pairs remain blank. Snapshot=last requested time; summary uses requested window only.',
        temporal_sensitivity='Repeat deterministic construction at every integer contact in inherited bracket, plus30. This is an uncertainty diagnostic, not independent repeats or temporal GT. Do not choose best availability. Legacy025 precision remains unknown.',
        phase_and_future='Only044/056 have inherited early point-end brackets checked here; other six are provisional service-aligned candidate inputs, not confirmed pre-outcome samples. No camera-exit/score-update time substitutes for point end. Underlying tracking may use future observations, so all processing is offline.',
        feature_isolation='Outcomes, source frame, absolute time, group, anchor status, paths, whole-rally duration and coverage are metadata only. No model or outcome association is fitted.',
        ball='No new ball positions or labels; ball comparison still limited to two same-outcome assisted examples. Do not encode missing ball datasets as physical absence.',
        label_provenance='Existing latest human camera-side labels, including025 near correction; team labels not independently confirmed. Pair swapping cannot fix wrong-person/wrong-side assignments.',
        split='One candidate per row; preserve point groups and match grouping. Eight rows are not independent frames or sufficient external performance evidence.',
        no_new_human_annotation=True,canonical_tracking_changed=False,protected_input_hashes=protected,
        script_sha256=sha(Path(__file__))))
    shutil.copy2(Path(__file__), OUT/'executed_source.py')
    frame_rows=[]; snapshots=[]; summaries=[]; metadata=[]; outcomes=[]; sensitivity=[]; observations=[]
    swaps=0
    for r in selected:
        cid=r['rally_candidate_id'];tag=cid[-3:];a=anchors[tag]; parts=json.loads(r['player_source_parts_json'])
        ps=players_from_parts(parts); t0=a['last']+30; times=list(range(t0,t0+61,2)); fs=features(cid,t0,ps)
        # Reuse the previous summary implementation, passing no ball observations; discard ball summary.
        summary_input=[dict(x,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for x in fs]
        snap,summ,_=summarize_prefix(tag,summary_input)
        snapshots.append(snap);summaries.append(summ);frame_rows.extend(fs)
        for f in times:
            players=ps.get(f,{})
            for slot in SLOTS:
                pr=players.get(slot)
                observations.append(dict(rally_candidate_id=cid,source_frame=f,relative_source_frame=f-t0,player_slot=slot,
                    observation_present=int(pr is not None),bottom_center_x='' if pr is None else pr['bottom_center_x'],
                    bottom_center_y='' if pr is None else pr['bottom_center_y'],track_id='' if pr is None else pr['track_id'],
                    position_source='' if pr is None else pr['position_source'],identity_status='' if pr is None else pr.get('identity_status',''),
                    missing_reason='' if pr is not None else 'before_player_export_start' if f<min(ps) else 'no_exported_observation_not_physical_absence'))
            for sn,sf in itertools.product([False,True],repeat=2):
                swapped={}
                for slot,pr in players.items():
                    side=slot.split('_')[0]; pair=[s for s in SLOTS if s.startswith(side+'_')]
                    target=pair[1-pair.index(slot)] if (sn if side=='near' else sf) else slot
                    swapped[target]=pr
                assert pair_features(players)==pair_features(swapped);swaps+=1
        augmented=dict(ps);augmented[times[-1]+2]={s:dict(bottom_center_x='1',bottom_center_y='1') for s in SLOTS}
        assert features(cid,t0,augmented)==fs
        assert latest[cid]['value']==labels[cid]['winner_camera_side']==r['winner_camera_side']
        assert a['end_first'] is None or times[-1]<a['end_first']
        metadata.append(dict(rally_candidate_id=cid,point_group_id=r['point_group_id'],point_group_status=r['point_group_status'],
            anchor_stratum=a['stratum'],anchor_first_source_frame=a['first'],anchor_last_source_frame=a['last'],
            input_first_source_frame=t0,input_end_source_frame=times[-1],requested_times=len(times),prefix_span_seconds=60/FPS,
            four_observed_times=sum(len(ps.get(f,{}))==4 for f in times),near_pair_observed_times=sum(x['near_pair_valid'] for x in fs),
            far_pair_observed_times=sum(x['far_pair_valid'] for x in fs),export_first_frame=min(ps),export_last_frame=max(ps),
            times_before_export_start=sum(f<min(ps) for f in times),
            inherited_end_bracket_available=int(a['end_first'] is not None),earliest_point_end_candidate=a['end_first'] or '',
            input_before_inherited_end_bracket=int(a['end_first'] is not None and times[-1]<a['end_first']),
            point_end_timing_human_confirmed=0,independent_identity_or_position_accuracy_established=0,
            anchor_source=a['source'],anchor_evidence=a['evidence'],validity_evidence=a['validity'],player_source_parts_json=json.dumps(parts)))
        outcomes.append(dict(rally_candidate_id=cid,point_group_id=r['point_group_id'],winner_camera_side=r['winner_camera_side'],human_review_id=latest[cid]['review_id']))
        variants=[]
        for contact in range(a['first'],a['last']+1):
            vf=features(cid,contact+30,ps)
            _,vs,_=summarize_prefix(tag,[dict(x,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for x in vf]);variants.append(vs)
        for feature in [x for x in summ if x!='rally_candidate_id']:
            values=[float(v[feature]) for v in variants if v[feature]!='']
            sensitivity.append(dict(rally_candidate_id=cid,feature=feature,bracket_variant_count=len(variants),available_variant_count=len(values),
                value_min=min(values) if values else '',value_max=max(values) if values else '',range_norm=max(values)-min(values) if values else '',
                uncertainty_scope='legacy_precision_unknown_nominal_only' if tag=='025' else 'inherited_AI_contact_bracket_only_not_accuracy'))
    for name,rows in [('requested_player_observations.csv',observations),('prefix_frame_features.csv',frame_rows),
            ('snapshot_player_features.csv',snapshots),('prefix_player_summary_features.csv',summaries),
            ('rally_metadata.csv',metadata),('human_outcomes.csv',outcomes),('contact_timing_sensitivity.csv',sensitivity)]:write(OUT/name,rows)
    dump(OUT/'feature_schema.json',dict(join_key='rally_candidate_id',snapshot_features=[k for k in snapshots[0] if k!='rally_candidate_id'],
        summary_features=[k for k in summaries[0] if k!='rally_candidate_id'],frame_time='relative_source_frame; not a fitted feature',
        missing='CSV blank; no interpolation, last-observation fill, or zero-coordinate substitution',
        metadata_files_not_model_features=['rally_metadata.csv','human_outcomes.csv','contact_timing_sensitivity.csv','requested_player_observations.csv']))
    assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(status='eight_candidate_player_inputs_constructed',candidates=len(metadata),requested_times=len(frame_rows),
        requested_slot_observations=len(observations),outcome_counts=dict(Counter(r['winner_camera_side'] for r in outcomes)),
        anchor_strata=dict(Counter(r['anchor_stratum'] for r in metadata)),
        inherited_early_end_bracket_available=sum(r['inherited_end_bracket_available'] for r in metadata),
        paired_slot_permutation_checks=swaps,future_observation_injection_checks=len(selected),
        model_fitting_performed=False,outcome_association_estimated=False,independent_tracking_accuracy_established=False,
        no_new_human_annotation=True,canonical_tracking_changed=False,
        artifact_hashes={p.name:sha(p) for p in OUT.glob('*.csv')})
    result['artifact_hashes']['feature_schema.json']=sha(OUT/'feature_schema.json')
    dump(OUT/'result.json',result)
    print(json.dumps(dict(result=result,metadata=[{k:r[k] for k in ['rally_candidate_id','four_observed_times','times_before_export_start','anchor_stratum']} for r in metadata]),ensure_ascii=False))

if __name__=='__main__':main()
