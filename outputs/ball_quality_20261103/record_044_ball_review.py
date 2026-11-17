"""Apply exactly three explicitly user-confirmed ball component choices."""
import csv,json
from pathlib import Path
from extract_044_inplay_ball_candidates import OUT,ROOT,read,sha

FRAMES=[773,784,1103]

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    folder=OUT/'user_reviewed';folder.mkdir(exist_ok=True)
    assert not (folder/'review.json').exists()
    protocol=json.loads((OUT/'protocol.json').read_text());cid=protocol['pilot']
    baseline=ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv'
    original=read(baseline);proposal_path=OUT/'substitution_proposals.csv'
    proposals={int(r['local_frame']):r for r in read(proposal_path)}
    assert sorted(t for t,r in proposals.items() if int(r['substitution_proposed']))==FRAMES
    assert sha(baseline)==protocol['baseline_sha256']
    candidates=read(OUT/'candidates.csv')
    review_id='20261103_044_ball_components_773_784_1103'
    reviewed=[]
    for t in FRAMES:
        p=proposals[t];r=original[t]
        assert any(int(c['local_frame'])==t and c['x_px']==p['proposed_x_px'] and c['y_px']==p['proposed_y_px'] for c in candidates)
        reviewed.append(dict(local_frame=t,source_frame=int(r['source_frame']),
            original_xy_px=[float(r['x_px']),float(r['y_px'])],
            reviewed_component_xy_px=[float(p['proposed_x_px']),float(p['proposed_y_px'])],
            component_rank=int(p['proposed_component_rank']),component_peak=float(p['proposed_component_peak']),
            candidate_is_ball=True,precise_pixel_center_reviewed=False))
    review=dict(review_id=review_id,recorded_date='2026-11-03',reviewer='user',ball_pilot_id=cid,
        rally_candidate_id='match1_rally_window_044',clock='legacy edited match1; clip local frame indices',
        question='この3箇所で、緑の丸は実際のボールを囲めていますか？',
        answer='この3箇所で、緑の丸は実際のボールを囲めている．',
        scope='The three displayed green candidates enclose the real ball. No full-track or precise-center confirmation.',
        candidates=reviewed,reviewed_video_sha256=sha(OUT/'substitution_review_h264.mp4'),
        proposal_csv_sha256=sha(proposal_path),frozen_evaluation_labels_modified=False,
        independent_localization_accuracy_measured=False)
    (folder/'review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    output=[]
    for t,r in enumerate(original):
        assert int(r['local_frame'])==t
        q=dict(r,raw_x_px=r['x_px'],raw_y_px=r['y_px'],raw_confidence=r['confidence'],
            raw_position_source=r['position_source'],component_rank='',component_peak='',
            coordinate_extractor='unchanged_original_model_location',
            ball_candidate_review='not_reviewed_by_this_confirmation',human_review_id='',
            precise_center_reviewed_here=0,interpolated_here=0)
        if t in FRAMES:
            p=proposals[t]
            q.update(x_px=p['proposed_x_px'],y_px=p['proposed_y_px'],
                confidence='',position_source='observed_heatmap_component_user_confirmed_ball',
                component_rank=p['proposed_component_rank'],component_peak=p['proposed_component_peak'],
                coordinate_extractor='half_peak_weighted_component_centroid',
                ball_candidate_review='user_confirmed_is_ball',human_review_id=review_id)
            # The raw detector's confidence belongs to its old selected location.
            # Store the alternative peak separately; it is not calibrated confidence.
        output.append(q)
    dest=folder/'predictions'/cid/'balltrack_predictions.csv';write_csv(dest,output)
    saved=read(dest);changed=[]
    for t,(before,after) in enumerate(zip(original,saved,strict=True)):
        if any(before[k]!=after[k] for k in before):changed.append(t)
        assert before['visible']==after['visible']
        assert after['raw_x_px']==before['x_px'] and after['raw_y_px']==before['y_px']
        assert after['raw_confidence']==before['confidence']
        if t not in FRAMES:assert all(before[k]==after[k] for k in before)
        else:
            assert after['x_px']==proposals[t]['proposed_x_px'] and after['y_px']==proposals[t]['proposed_y_px']
            assert after['confidence']=='' and after['human_review_id']==review_id
    assert changed==FRAMES
    prior=json.loads((ROOT/'play_start_audit/inventory_summary.json').read_text())
    for name,expected in prior['source_hashes'].items():assert sha(Path(name))==expected
    assert sha(baseline)==protocol['baseline_sha256']
    report=dict(review_id=review_id,full_clip_rows=len(saved),changed_local_frames=changed,
        other_rows_original_columns_unchanged=True,original_baseline_unchanged=True,
        visibility_unchanged=True,missing_outputs_added=0,interpolated_positions=0,
        source_review_tables_unchanged=True,precise_center_groundtruth_created=False,
        reviewed_output=str(dest.resolve()),reviewed_output_sha256=sha(dest),
        baseline_sha256=sha(baseline),review_sha256=sha(folder/'review.json'))
    (folder/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    override_path=ROOT/'reviewed_ball_overrides.json'
    override=json.loads(override_path.read_text()) if override_path.exists() else dict(scope='Explicitly reviewed per-clip overrides; unchanged detector baseline remains available',clips={})
    override['clips'][cid]=dict(rally_candidate_id='match1_rally_window_044',reviewed_output=str(dest.resolve()),
        reviewed_output_sha256=sha(dest),original_baseline=str(baseline.resolve()),
        reviewed_changed_local_frames=FRAMES,review_id=review_id,
        review_record=str((folder/'review.json').resolve()),automatic_filter_adopted=False,
        precise_center_or_full_track_accuracy_established=False)
    override_path.write_text(json.dumps(override,indent=2),encoding='utf-8')
    decision=json.loads((OUT/'decision.json').read_text())
    decision.update(status='three_user_confirmed_component_substitutions_applied_in_reviewed_output',
        human_ball_identity_review='user_confirmed_three_displayed_candidates_are_ball',
        reviewed_output=str(dest.resolve()),review_record=str((folder/'review.json').resolve()),
        next='Repair far-player initialization near effective serve; retain multi-frame ball switches and long gaps as unresolved.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
