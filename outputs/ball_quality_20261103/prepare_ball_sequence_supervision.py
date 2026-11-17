"""Export explicit candidate labels with exact four-frame TrackNet input views.

No dense ball centers, boxes or absent-ball masks are invented from circle
reviews. The old300-center table is audited, never imported as confirmed truth.
"""
import json
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha

OUT=ROOT/'ball_sequence_candidate_supervision'

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'manifest.json').exists()
    cp=ROOT/'ball_candidate_cross_rally/user_reviewed/all_reviewed_candidate_cases.csv';corpus=read(cp)
    assert len(corpus)==36 and sha(cp)==load(cp.parent/'corpus_manifest.json')['all_cases_csv_sha256']
    protected={str(p):sha(p) for p in [cp,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']}
    labels_path=SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv';legacy=read(labels_path)
    protected[str(labels_path)]=sha(labels_path)
    source_loader=SOURCE/'third_party/RacketVision/source/BallTrack/dataset/uball.py'
    source_review_ui=SOURCE/'scripts/39_review_balltrack_groundtruth_server.py'
    audit=dict(rows=len(legacy),visibility_counts=dict(Counter(r['truth_visibility'] for r in legacy)),
        visible_rows_with_center=sum(r['truth_visibility']=='visible' and bool(r['truth_x_px']) and bool(r['truth_y_px']) for r in legacy),
        nonblank_reviewer_ids=sum(bool(r['reviewer_id']) for r in legacy),
        source_sha256=sha(labels_path),review_ui=str(source_review_ui),review_ui_sha256=sha(source_review_ui),
        provenance_interpretation='Empty reviewer IDs do not prove no human review: the inspected review server save handler does not populate reviewer_id. Author/independence remain unverified.',
        dense_label_loader=str(source_loader),dense_label_loader_sha256=sha(source_loader),
        legacy_table_used_for_training_in_this_step=False,
        candidate_review_semantics='A circle encloses/does not enclose a real ball; not an exact center or full-frame physical visibility annotation')
    dump(OUT/'legacy_label_audit.json',audit)
    items=[];views=set();reviews={}
    inventory={r['ball_pilot_id']:r for r in read(ROOT/'play_start_audit/ball_pilot_start_inventory.csv')}
    for c in corpus:
        cid=c['ball_pilot_id'];t=int(c['local_frame']);x=float(c['x_px']);y=float(c['y_px'])
        rect=[0,0,1920,1080];view='full_frame'
        if c['inference_view']=='tiled2x':
            rp=Path(c['review_record']);review=reviews.setdefault(str(rp),load(rp))
            entries=review['confirmed_observations']+review['rejected_tiled_components']
            matches=[r for r in entries if int(r['local_frame'])==t and abs(float(r['x_px'])-x)<1e-7 and abs(float(r['y_px'])-y)<1e-7]
            assert len(matches)==1
            rect=matches[0]['component_tile_xywh'];view='tiled2x'
        left,top,width,height=rect
        assert left<=x<left+width and top<=y<top+height
        source=SOURCE/'data/balltrack/racketvision/match1'/cid
        frame_paths=[source/'frames'/f'{j:06d}.jpg' for j in range(t-3,t+1)]
        median=source/'runs/threshold030_median120/median.npz'
        assert t>=3 and all(p.is_file() for p in frame_paths) and median.is_file()
        assert int(c['source_frame'])==int(inventory[cid]['clip_first_source_frame'])+t
        query=[(x-left)*512/width,(y-top)*288/height]
        item=dict(case_id=c['case_id'],ball_pilot_id=cid,local_frame=t,source_frame=int(c['source_frame']),
            split_group=c['split_group'],candidate_is_ball=int(c['label']=='confirmed_ball_enclosed'),
            annotation_kind='candidate_encloses_ball_binary_only',precise_center_gt=False,full_frame_visibility_gt=False,
            view_name=view,view_xywh=rect,candidate_source_xy=[x,y],query_network_xy=query,
            frames_local=list(range(t-3,t+1)),frame_paths=[str(p) for p in frame_paths],
            frame_sha256=[sha(p) for p in frame_paths],median_path=str(median),median_sha256=sha(median),
            review_record=c['review_record'],review_sha256=sha(Path(c['review_record'])))
        items.append(item);views.add((cid,t,tuple(rect)))
    assert Counter(r['candidate_is_ball'] for r in items)=={1:21,0:15}
    # No view's positive and negative labels are silently merged into a frame label.
    assert len({r['case_id'] for r in items})==36
    dump(OUT/'sequence_cases.json',items)
    index=[dict(case_id=r['case_id'],ball_pilot_id=r['ball_pilot_id'],local_frame=r['local_frame'],
        candidate_is_ball=r['candidate_is_ball'],split_group=r['split_group'],view_name=r['view_name'],
        view_xywh=json.dumps(r['view_xywh']),query_network_xy=json.dumps(r['query_network_xy']),
        source_frame=r['source_frame'],annotation_kind=r['annotation_kind']) for r in items]
    write(OUT/'candidate_index.csv',index)
    assert all(sha(Path(p))==h for p,h in protected.items())
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    manifest=dict(cases=36,real=21,false=15,source_clips=4,unique_frame_view_inputs=len(views),
        full_frame_candidates=sum(r['view_name']=='full_frame' for r in items),tiled_candidates=sum(r['view_name']=='tiled2x' for r in items),
        sequence='median plusBGR t-3..t; image crop from view_xywh then512x288resize; input ends at labelled frame',
        supervision='Binary candidate classification only; all unsupervised pixels remain unknown',
        labels_not_created=['precise_ball_center','ball_bounding_box','full_frame_absence','dense_heatmap_target','interpolated_position'],
        source_annotations_and_tracking_inputs_unchanged=True,sequence_cases_sha256=sha(OUT/'sequence_cases.json'),
        source_hashes=protected,dataset_use='Candidate-level temporal feature/head experiment; not directly compatible with a dense Gaussian heatmap training loader',
        old300_labels_imported_as_confirmed_training_targets=False,independent_accuracy_established=False)
    dump(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='source_hashes'},indent=2))

if __name__=='__main__':main()
