"""Suggest alternative observed components for isolated baseline excursions.

Never deletes an observation or interpolates a coordinate. Suggestions remain
unadopted until visual review; temporal support is not ball ground truth.
"""
import csv,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from extract_044_inplay_ball_candidates import OUT,ROOT,SOURCE,read,sha

PARAMS=dict(raw_center_residual_min_px=64.,alternate_center_residual_max_px=24.,
    alternate_peak_min=.15,anchor_endpoint_distance_max_px=60.,
    competing_alternate_distance_margin_px=8.,anchors='raw observations exactly t-1 and t+1',
    scope='existing candidate in-play interval only',automatic_adoption=False)

def proposals(raw,candidates,start,end):
    by=defaultdict(list)
    for c in candidates:by[int(c['local_frame'])].append(c)
    xy=lambda r:np.array([float(r['x_px']),float(r['y_px'])])
    changes={};decisions=[]
    for t in range(start+1,end):
        if not all(int(raw[k]['visible']) for k in [t-1,t,t+1]):continue
        left,right=xy(raw[t-1]),xy(raw[t+1]);mid=(left+right)/2
        residual=float(np.linalg.norm(xy(raw[t])-mid))
        if residual<PARAMS['raw_center_residual_min_px']:continue
        separation=float(np.linalg.norm(right-left))
        reason='no_supported_alternative';possible=[]
        if separation<=PARAMS['anchor_endpoint_distance_max_px']:
            for c in by[t]:
                error=float(np.linalg.norm(xy(c)-mid))
                if float(c['peak'])>=PARAMS['alternate_peak_min'] and error<=PARAMS['alternate_center_residual_max_px']:
                    possible.append((error,c))
            possible.sort(key=lambda p:p[0])
            if possible and (len(possible)==1 or possible[1][0]-possible[0][0]>=PARAMS['competing_alternate_distance_margin_px']):
                error,chosen=possible[0];reason='proposed_observed_component_substitution'
                changes[t]=dict(chosen,raw_residual_px=residual,alternate_residual_px=error,
                    anchor_first_local_frame=t-1,anchor_last_local_frame=t+1,
                    selected_component_source='unchanged_model_observed_heatmap_component',independent_ball_confirmation=False)
            elif possible:reason='ambiguous_alternatives_preserve_raw'
        else:reason='anchors_too_far_apart_preserve_raw'
        decisions.append(dict(local_frame=t,raw_residual_px=residual,anchor_endpoint_distance_px=separation,
            decision=reason,alternatives_within_gate=len(possible)))
    return changes,decisions

def main():
    protocol=json.loads((OUT/'protocol.json').read_text());start,end=protocol['local_frames'];cid=protocol['pilot']
    bp=ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv'
    raw=read(bp);cp=OUT/'candidates.csv';candidates=read(cp)
    changes,decisions=proposals(raw,candidates,start,end);rows=[]
    for t in range(start,end+1):
        c=changes.get(t);r=raw[t]
        rows.append(dict(r,proposed_x_px=c['x_px'] if c else r['x_px'],proposed_y_px=c['y_px'] if c else r['y_px'],
            proposed_visible=r['visible'],substitution_proposed=int(c is not None),
            proposed_component_rank=c['rank'] if c else '',proposed_component_peak=c['peak'] if c else '',
            proposed_position_source='observed_alternative_component_unreviewed' if c else 'unchanged_raw_model_output',
            proposal_review_status='pending_visual_review' if c else 'no_change',interpolated=0))
    with (OUT/'substitution_proposals.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Verification checks origin and preservation, not correctness.
    original_fields=list(raw[0]);assert len(rows)==end-start+1
    for r in rows:
        t=int(r['local_frame']);assert all(r[k]==raw[t][k] for k in original_fields)
        assert int(r['proposed_visible'])==int(raw[t]['visible']) and r['interpolated']==0
        if t in changes:
            assert any(float(c['x_px'])==float(r['proposed_x_px']) and float(c['y_px'])==float(r['proposed_y_px']) for c in candidates if int(c['local_frame'])==t)
        else:assert str(r['proposed_x_px'])==raw[t]['x_px'] and str(r['proposed_y_px'])==raw[t]['y_px']
    assert sha(bp)==protocol['baseline_sha256']
    # Report the correct four-player file separately from the old two-near-player export.
    person=ROOT.parent/'frontcourt_handoff_20260927/full_rally_044/bytetrack_four_player_reconnection.csv'
    pr=[r for r in read(person) if int(raw[start]['source_frame'])<=int(r['source_frame'])<=int(raw[end]['source_frame'])]
    counts=Counter(int(r['source_frame']) for r in pr)
    person_summary=dict(path=str(person),coordinate_scale_to_cached_1920x1080=.5,
        rows=len(pr),frames_with_four_candidate_rows=sum(n==4 for n in counts.values()),
        positions_and_continuous_identity_unverified=True,
        slots={slot:dict(Counter(r['position_source'] for r in pr if r['player_slot']==slot)) for slot in {r['player_slot'] for r in pr}},
        note='Do not use bytetrack_selected_four.csv as all-four tracking: it only contains two near-player slots in this interval.')
    report=dict(parameters=PARAMS,script_sha256=sha(Path(__file__)),candidate_file_sha256=sha(cp),
        baseline_file_sha256=sha(bp),scope_frames=[start,end],baseline_output_frames=sum(int(raw[t]['visible']) for t in range(start,end+1)),
        candidate_frames_with_components=len({int(c['local_frame']) for c in candidates}),
        suggested_changes={str(t):c for t,c in changes.items()},decisions=decisions,
        changed_frames=len(changes),missing_outputs_added=0,raw_observations_deleted=0,
        person_input_audit=person_summary,automatic_adoption=False,independent_accuracy_claim=False,
        limitation='Single-clip development. A temporal alternative may also be wrong; no precision/recall measured.')
    (OUT/'substitution_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(original_columns_and_source_files_unchanged=True,
        alternative_centers_observed_in_same_frame=True,no_missing_outputs_added=True,no_interpolation=True,
        accuracy_established=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(changes={str(t):[c['x_px'],c['y_px'],c['rank']] for t,c in changes.items()},decisions=decisions,person_audit=person_summary),indent=2))

if __name__=='__main__':main()
