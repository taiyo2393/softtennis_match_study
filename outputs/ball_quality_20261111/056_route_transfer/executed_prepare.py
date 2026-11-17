"""Frozen-method transfer to second pre-existing assisted rally, not independent accuracy."""
from pathlib import Path
from collections import defaultdict
import math,shutil
import cv2,numpy as np
import build_044_spatiotemporal_pilot as b

OUT=b.HERE/'056_route_transfer'
BALL=b.PRIOR/'rally_repair_056/completed_review'
FRAMES=BALL.parent/'frames'
PERSON=b.HERE.parent/'tracking_quality_20261102/players_056_appearance_motion_guarded/selected_four.csv'
OFFSET=154078
TIMES=list(range(154080,154881,2))

def main():
    assert not OUT.exists();OUT.mkdir()
    files=[Path(__file__),PERSON,b.CAL,BALL/'assisted_states_sampled.csv',BALL/'result.json',BALL/'verification.json',BALL/'cases.json',BALL/'protocol.json',
        b.HERE/'044_estimated_ball_routes/publication_hashes.json',b.HERE/'044_estimated_ball_routes/protocol.json',
        b.HERE/'044_estimated_ball_routes/court_constrained_trial_v2/protocol.json',b.HERE/'estimate_044_far_bounces.py',b.HERE/'constrain_044_far_bounce_fit.py',
        b.LABEL,b.HERE.parent/'ball_quality_20261110/current22_player_inputs/publication_hashes.json']
    b.dump(OUT/'protocol.json',dict(date_jst='2026-11-11',case='match1_rally_window_056',selection='Previously completed second assisted rally056; retain full inherited contact/end context, not a new good-looking tracking window.',
        source_clock='legacy_edited_match1',source_offset=OFFSET,source_images=str(FRAMES),display_frames=TIMES,
        ball='376previous assisted reviewed times154088..154838,334positions/42unavailable. Do not revise centers or unknown labels. Not raw automation or blind GT.',
        people='Existing guarded selected_four path. Same calibration and bbox-bottom proxy convention. No new tracking or per-person manual rebinding.',
        inherited_events='Effective opening serve valid,near winner,two-bounce ending are old human facts. Contact154081..154086/end154840..154880 remain AI-provisional. Do not reask old facts or infer exact timings.',
        frozen_methods='Same45degree direction turn rule,5visible samples,stride2,>=0.5px/sourceframe,12frame separation. Same fixed24frame bounce-fit neighborhood,continuous piecewise x-linear/y-quadratic/sign constraints,integer candidate times. Same court-polygon constraint if needed; retain unconstrained failures. Same <=12frame within-flight interpolation; never bridge contacts/bounces. Same blue schematic-not3D route display.',
        event_scope='AI chronological review needed for this clip; mark new stroke/bounce proposals separately. Do not force a bounce for volleys or copy044hitter order/court sides. No new independent accuracy estimate.',
        policy_fixed_before_new056event_or_model_outputs=True,protected_input_hashes={str(p):b.sha(p) for p in files}))
    result=b.load(BALL/'result.json');assert all(b.sha(BALL/n)==h for n,h in result['export_hashes'].items())
    by=b.grouped(b.read(PERSON));balls={int(r['source_frame']):r for r in b.read(BALL/'assisted_states_sampled.csv')};H=b.load(b.CAL)['image_to_court_homography']
    people=[];ballrows=[];states=[];images=[]
    for f in TIMES:
        image=FRAMES/f'{f-OFFSET:06d}.jpg';assert image.exists();images.append(dict(source_frame=f,path=str(image),sha256=b.sha(image)))
        for r in by[f]:
            u,v=b.project(H,float(r['bottom_center_x']),float(r['bottom_center_y']))
            people.append(dict(source_frame=f,player_slot=r['player_slot'],track_id=r['track_id'],image_x_px=float(r['bottom_center_x'])/2,image_y_px=float(r['bottom_center_y'])/2,
                x1_px=float(r['x1'])/2,y1_px=float(r['y1'])/2,x2_px=float(r['x2'])/2,y2_px=float(r['y2'])/2,
                court_proxy_x=u,court_proxy_y=v,position_source=r['position_source'],identity_status=r['identity_status'],independent_position_validated=0))
        z=balls.get(f);visible=z is not None and z['observation_present']=='1'
        ballrows.append(dict(source_frame=f,annotation_available=int(z is not None),observation_present=int(visible),image_x_px=z['x_px'] if visible else '',image_y_px=z['y_px'] if visible else '',
            coordinate_source=z['coordinate_source'] if z else 'not_annotated',court_x='',court_y='',independent_position_GT=0))
        states.append(dict(source_frame=f,person_count=len(by[f]),ball_observation_present=int(visible),annotation_available=int(z is not None)))
    b.write(OUT/'player_court_proxies.csv',people);b.write(OUT/'ball_image_observations.csv',ballrows);b.write(OUT/'frame_states.csv',states);b.dump(OUT/'source_image_manifest.json',images)
    # Exact original turn rule, without parameter selection on056.
    points={r['source_frame']:np.array([float(r['image_x_px']),float(r['image_y_px'])]) for r in ballrows if r['observation_present']}
    proposals=[]
    for f in sorted(points):
        if not all(f+d in points for d in [-4,-2,0,2,4]):continue
        vin=(points[f]-points[f-4])/4;vout=(points[f+4]-points[f])/4;norms=[np.linalg.norm(v) for v in [vin,vout]]
        if min(norms)<.5:continue
        angle=math.degrees(math.acos(float(np.clip(np.dot(vin,vout)/np.prod(norms),-1,1))))
        if angle>=45:proposals.append(dict(source_frame=f,turn_degrees=angle,ball_x=float(points[f][0]),ball_y=float(points[f][1]),event_status='direction_change_only_not_hit_or_bounce_GT'))
    kept=[]
    for r in sorted(proposals,key=lambda r:(-r['turn_degrees'],r['source_frame'])):
        if all(abs(r['source_frame']-v['source_frame'])>=12 for v in kept):kept.append(r)
    kept.sort(key=lambda r:r['source_frame'])
    for i,r in enumerate(kept,1):r['candidate_id']=f'C{i:02}'
    b.write(OUT/'direction_change_candidates.csv',kept)
    # Fixed chronological grid, full court width, no selecting only successful turns.
    lookup={r['source_frame']:r for r in ballrows};tiles=[]
    for f in range(154080,154881,12):
        im=cv2.imread(str(FRAMES/f'{f-OFFSET:06d}.jpg'));assert im is not None;r=lookup[f]
        if r['observation_present']:cv2.circle(im,(round(float(r['image_x_px'])),round(float(r['image_y_px']))),9,(80,255,100),2)
        crop=cv2.resize(im[430:980,0:1920],(768,220));h=np.zeros((34,768,3),np.uint8)
        cv2.putText(h,f'F{f} / local{f-OFFSET} / {(f-154086)/b.FPS:+.2f}s',(8,24),0,.58,(255,255,255),1,cv2.LINE_AA);tiles.append(np.vstack([h,crop]))
    for i in range(0,len(tiles),12):
        batch=tiles[i:i+12]
        while len(batch)<12:batch.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT/f'chronology_{i//12+1:02}.jpg'),np.vstack([np.hstack(batch[j:j+2]) for j in range(0,12,2)]))
    b.dump(OUT/'preparation_result.json',dict(display_frames=len(TIMES),person_observations=len(people),four_candidate_frames=sum(r['person_count']==4 for r in states),
        ball_annotated_times=sum(r['annotation_available'] for r in ballrows),ball_positions=sum(r['observation_present'] for r in ballrows),direction_change_candidates=len(kept),
        new_human_labels=0,independent_accuracy_established=False))
    shutil.copy2(Path(__file__),OUT/'executed_prepare.py');print(b.load(OUT/'preparation_result.json'),flush=True)

if __name__=='__main__':main()
