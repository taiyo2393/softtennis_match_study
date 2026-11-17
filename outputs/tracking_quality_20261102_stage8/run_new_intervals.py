"""Apply the unchanged reviewed stage7 method to three additional intervals."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXPERIMENTS=ROOT.parent
sys.path.insert(0,str(EXPERIMENTS/'tracking_quality_20261102_stage7'))
import tracklet_pair_tracking as g
import run_temporal_experiment as initialization

spec=importlib.util.spec_from_file_location('fixed_detector',EXPERIMENTS/'tracking_quality_20261102_stage3/validate_new_rallies.py')
detector=importlib.util.module_from_spec(spec);spec.loader.exec_module(detector)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((EXPERIMENTS/'tracking_quality_20261102_stage7/players_002_global/protocol.json').read_text(encoding='utf-8'))
    assert g.P==baseline['parameters']
    assert sha(Path(g.__file__))==baseline['code_sha256']
    ready=g.a.read(g.a.SOURCE/'data/segments/analysis_ready_main_camera_segments.csv')
    intervals=[]
    for tag,start in [('003',4201),('007',15552),('025',60006)]:
        matches=[r for r in ready if r['video_id']=='match1' and r['usable_for_analysis'].lower()=='true'
                 and int(r['start_source_frame'])<=start<=int(r['end_source_frame'])]
        assert len(matches)==1
        end=min(start+599,int(matches[0]['end_source_frame']))
        intervals.append(dict(tag=tag,first_frame=start,last_frame=end,frames=end-start+1,
                             boundary_status='candidate_interval_not_human_confirmed_full_rally'))
    files=[Path(g.__file__),Path(initialization.__file__),Path(detector.__file__),
           Path(g.t.previous.__file__),Path(g.a.__file__),ROOT/'run_new_intervals.py']
    plan=dict(intervals=intervals,parameters=g.P,source_sha256={str(p):sha(p) for p in files},
              max_source_frames=600,settings_changes_allowed=False,
              scope='Additional same-match intervals; previous snapshots existed, not independent match validation')
    protocol=ROOT/'transfer_protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text(encoding='utf-8'))==plan
    else:protocol.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    rawroot=ROOT/'detector_runs'
    g.a.BASE=rawroot
    g.t.previous.OUT=ROOT/'crop_runs';g.t.previous.OUT.mkdir(exist_ok=True)
    initialization.OUT=ROOT/'tracking_quality_20261102_stage4';initialization.OUT.mkdir(exist_ok=True)
    g.OUT=ROOT/'tracking_quality_20261102_stage7';g.OUT.mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'ultralytics_config')
    reports=[]
    for interval in intervals:
        tag=interval['tag']
        status=detector.detect(interval,g.t.selector)
        folder=rawroot/f'full_rally_{tag}'
        # Crop augmentation only needs the interval, even if first-frame seeding fails.
        config=folder/'four_player_selection_summary.json'
        if not config.exists():
            config.write_text(json.dumps([dict(seed_frame=interval['first_frame'],
                analysis_ready_end_frame=interval['last_frame'])],indent=2),encoding='utf-8')
        crop=g.t.previous.OUT/f'augmented_detections_{tag}.csv'
        if not crop.exists():crop=g.t.previous.augment(tag)
        init=initialization.process(interval,crop)
        if init['initialized_frame'] is None:
            status['global_status']='withheld_no_temporal_initialization'
        else:
            output=g.OUT/f'players_{tag}_global'
            if not (output/'summary.json').exists():
                sys.argv=[g.__file__,'--tag',tag];g.main()
            status['global_status']='candidate_created'
            status['global_summary']=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            assert json.loads((output/'protocol.json').read_text(encoding='utf-8'))==baseline
        reports.append(status)
        (ROOT/'transfer_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    for path,digest in plan['source_sha256'].items():assert sha(Path(path))==digest
    print(json.dumps([dict(tag=r['tag'],status=r['global_status']) for r in reports]),flush=True)


if __name__=='__main__':main()
