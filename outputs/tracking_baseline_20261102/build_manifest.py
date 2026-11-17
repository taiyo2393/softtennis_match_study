"""Index reviewed tracking artifacts without converting video reviews into GT."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
reviews={
 '002':BASE/'tracking_quality_20261102_stage7/players_002_global/human_video_review.json',
 '003':BASE/'tracking_quality_20261102_stage9/players_003_temporal/human_video_review.json',
 '005':BASE/'tracking_quality_20261102_stage7/players_005_global/human_video_review.json',
 '007':BASE/'tracking_quality_20261102_stage8/tracking_quality_20261102_stage7/players_007_global/human_video_review.json',
 '025':BASE/'tracking_quality_20261102_stage10/experiment/tracking_quality_20261102_stage7/players_025_global/human_video_review.json',
 '048':BASE/'tracking_quality_20261102_stage6/players_048_global/human_video_review.json'}
runs=[]
for tag,reviewpath in reviews.items():
    folder=BASE/'tracking_quality_20261102_stage10/experiment/tracking_quality_20261102_stage7'/f'players_{tag}_global'
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    review=json.loads(reviewpath.read_text(encoding='utf-8'))
    runs.append(dict(candidate_id=tag,video=str((folder/'global_review_h264.mp4').resolve()),
        far_positions_csv=str((folder/'far_pair_path.csv').resolve()),
        person_correspondence_review=str(reviewpath.resolve()),review_scope=review['scope'],
        current_output_status='path_reviewed_directly_or_preserved_from_reviewed_version',
        summary=summary))
report=dict(status='provisional_research_baseline',runs=runs,
    method='YOLO11m + ByteTrack/crop candidates + temporal initialization + offline joint far-pair search + smooth tracklet/short-gap consistency',
    limitations=['Same-match short intervals only','Video-level correspondence confirmation, not independent per-frame ID accuracy',
                 'Missing frames not filled','BBox bottom positions approximate; metric ground-plane position accuracy not established',
                 'Far-pair algorithm and reused near tracks are separate branches; some near-side gaps remain'],
    code= str((BASE/'tracking_quality_20261102_stage10/gap_pair_tracking.py').resolve()),
    checks=str((BASE/'tracking_quality_20261102_stage10/regression_and_integrity_checks.json').resolve()))
ROOT.mkdir(exist_ok=True)
(ROOT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Indexed six reviewed or output-equivalent tracking artifacts; review scopes retained.')
