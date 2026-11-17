"""Validate audit media against source and join AI phase evidence as separate metadata."""
import csv,hashlib,json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'ball_quality_20261110/prefix_phase_audit'
INPUT=ROOT.parent/'ball_quality_20261109/rally_player_prefix_expansion'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    assert not (OUT/'result.json').exists(),'Preserve audit result.'
    protocol=load(OUT/'protocol.json');manifest=load(OUT/'evidence_manifest.json');visual=load(OUT/'visual_findings.json')
    meta=read(INPUT/'rally_metadata.csv');by_meta={r['rally_candidate_id']:r for r in meta}
    findings={r['candidate_id']:r for r in visual['findings']};assert len(findings)==len(manifest)==6
    assert set(findings)=={r['candidate_id'] for r in manifest}
    cap=cv2.VideoCapture(protocol['source_video']);assert cap.isOpened();checks=0;rendered=0
    for e in manifest:
        cid=e['candidate_id'];m=by_meta[cid];tag=cid[-3:];fnd=findings[cid]
        assert e['requested_input_range']==[int(m['input_first_source_frame']),int(m['input_end_source_frame'])]
        assert e['rendered_frames']==e['last_source_frame']-e['first_source_frame']+1
        assert e['input_grid_source_frames']==list(range(e['requested_input_range'][0],e['requested_input_range'][1]+1,4))
        assert sha(Path(e['clip']))==e['clip_sha256']
        clip=cv2.VideoCapture(e['clip']);assert int(clip.get(cv2.CAP_PROP_FRAME_COUNT))==e['rendered_frames']
        assert (int(clip.get(3)),int(clip.get(4)))==(960,576)
        assert abs(clip.get(5)-protocol['fps'])<.001
        decoded=0
        while True:
            ok,im=clip.read()
            if not ok:break
            assert im.shape[:2]==(576,960);decoded+=1
        assert decoded==e['rendered_frames'];clip.release();rendered+=decoded
        for name,h in e['image_hashes'].items():assert sha(OUT/tag/name)==h
        for f in [e['requested_input_range'][0],e['requested_input_range'][1],e['overview_source_frames'][0],e['overview_source_frames'][-1]]:
            cap.set(1,f);ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
            image=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA)
            ok,buf=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
            assert hashlib.sha256(buf.tobytes()).hexdigest()==e['image_hashes'][f'F{f}.jpg'];checks+=1
        assert all(e['requested_input_range'][1]<f<=e['last_source_frame'] for f in fnd['later_play_evidence_frames'])
        assert all(f in e['overview_source_frames'] for f in fnd['later_play_evidence_frames'])
        assert fnd['input_phase_status']=='ai_visual_sequence_supported'
        assert not fnd['exact_service_validity_human_confirmed_by_this_audit'] and not fnd['exact_point_end_confirmed']
    cap.release()
    detail=load(OUT/'025/legacy_contact_detail_manifest.json');assert sha(OUT/'025/legacy_contact_detail.jpg')==detail['sha256']
    # Recreate every detail crop; preserve rectangular aspect ratio exactly.
    assert detail['crop_xyxy']==[2500,650,3400,1900] and detail['output_wh']==[360,500]
    assert 900/1250==360/500
    cap=cv2.VideoCapture(protocol['source_video']);cells=[]
    for f in detail['frames']:
        cap.set(1,f);ok,im=cap.read();assert ok
        crop=cv2.resize(im[650:1900,2500:3400],(360,500),interpolation=cv2.INTER_AREA)
        header=np.zeros((30,360,3),dtype=np.uint8)
        cv2.putText(header,f'025 F{f} {(f-59976)/59.94005994:+.3f}s',(8,22),0,.6,(255,255,255),1)
        cells.append(np.vstack([header,crop]))
    while len(cells)%4:cells.append(np.zeros_like(cells[0]))
    image=np.vstack([np.hstack(cells[i:i+4]) for i in range(0,len(cells),4)])
    ok,buf=cv2.imencode('.jpg',image);assert ok and hashlib.sha256(buf.tobytes()).hexdigest()==detail['sha256'];cap.release()
    assert all(sha(Path(p))==h for p,h in protocol['protected_input_hashes'].items())
    outcomes={r['rally_candidate_id']:r for r in read(INPUT/'human_outcomes.csv')};rows=[]
    for m in meta:
        cid=m['rally_candidate_id'];f=findings.get(cid)
        rows.append(dict(rally_candidate_id=cid,point_group_id=m['point_group_id'],winner_camera_side=outcomes[cid]['winner_camera_side'],
            existing_human_side_review_id=outcomes[cid]['human_review_id'],
            input_first_source_frame=m['input_first_source_frame'],input_end_source_frame=m['input_end_source_frame'],
            input_phase_evidence='new_AI_chronological_image_review' if f else 'inherited_provisional_service_and_end_brackets',
            input_phase_status=f['input_phase_status'] if f else 'within_inherited_candidate_main_play',
            human_service_validity_previously_confirmed=int(m['anchor_stratum']=='human_valid_service_ai_timing'),
            exact_contact_time_confirmed_by_this_audit=0,independent_phase_ground_truth_added=0,
            independent_tracking_accuracy_established=0,model_inputs_unchanged=1,
            source_findings_json=str(OUT/'visual_findings.json') if f else m['anchor_source']))
    write(OUT/'eight_candidate_phase_evidence.csv',rows)
    dump(OUT/'verification.json',dict(status='passed',source_frame_JPEG_recreations=checks,
        continuous_clip_frames_decoded=rendered,continuous_clips=6,detail_crop_frames_recreated=len(detail['frames']),
        image_and_clip_hashes_match=True,protected_previous_outputs_unchanged=True,
        note='Verifies evidence generation and linkage; does not independently validate AI visual phase decisions.'))
    dump(OUT/'result.json',dict(status='six_AI_phase_audits_recorded',ai_sequence_supported=6,
        visible_input_boundary_problem_found=0,independent_human_phase_reviews_added=0,
        previous_human_valid_service_labels_preserved=2,exact_event_times_added=0,model_fit=False,
        prior_feature_outputs_changed=False,new_human_annotation_requested=False,
        conclusion='All six fixed early windows are visually consistent with ongoing service-led play. Retain as AI-supported exploratory input metadata; do not promote to independent phase/position GT.',
        special_case='025 low-to-high service motion shows why overhead-only start cues are insufficient; style remains AI observation.',
        file_hashes={p.name:sha(p) for p in [OUT/'visual_findings.json',OUT/'evidence_manifest.json',OUT/'eight_candidate_phase_evidence.csv',OUT/'verification.json']}))
    print(json.dumps(load(OUT/'verification.json'),ensure_ascii=False));print('Six AI phase findings joined; no previous input or human label overwritten.')

if __name__=='__main__':main()
