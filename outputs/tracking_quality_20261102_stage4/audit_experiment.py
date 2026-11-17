"""Integrity, candidate availability and cross-version identity disagreements."""
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from temporal_tracker import a,PARAMETERS

OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'tracking_quality_20261102_stage3'


def grouped(rows):
    result = defaultdict(list)
    for r in rows:
        result[int(r['source_frame'])].append(r)
    return result


def main():
    protocol = json.loads((OUT/'experiment_protocol.json').read_text(encoding='utf-8'))
    results = json.loads((OUT/'experiment_results.json').read_text(encoding='utf-8'))
    baseline = {r['tag']:r for r in json.loads((OLD/'validation_results.json').read_text(encoding='utf-8'))}
    for path,digest in protocol['sources_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    report = dict(scope='Development candidates; availability and disagreements, not identity accuracy',
                  comparison=[],reacquisitions=[],media=[])
    differences = []
    for summary in results:
        tag = summary['tag']
        folder = OUT/f'players_{tag}_temporal'
        rows = a.read(folder/'selected_four.csv')
        frames = a.read(folder/'frame_status.csv')
        raw = a.read(Path(summary['input_csv']))
        assert hashlib.sha256(Path(summary['input_csv']).read_bytes()).hexdigest() == summary['input_sha256']
        events = json.loads((folder/'events.json').read_text(encoding='utf-8'))
        byframe,rawframe = grouped(rows),grouped(raw)
        assert len(frames) == summary['frames']
        for row in rows:
            frame = int(row['source_frame'])
            assert frame >= summary['initialized_frame']
            assert row['position_source'] == 'detected_box'
            assert row['identity_status'] == 'candidate_not_ground_truth'
            assert row['identity_review_status'] == 'requires_identity_review'
            matches = [r for r in rawframe[frame] if r['track_id'] == row['track_id']]
            assert len(matches) == 1
            assert all(row[k] == matches[0][k] for k in ('x1','y1','x2','y2','bottom_center_x','bottom_center_y'))
        for rs in byframe.values():
            assert len({r['player_slot'] for r in rs}) == len(rs) <= 4
            assert len({r['track_id'] for r in rs}) == len(rs)
        for e in events:
            if e['event'] == 'initialized':
                assert e['frame']-e['window_start']+1 == PARAMETERS['initialization_source_frames']
            else:
                assert e['confirmation_frames'] >= PARAMETERS['recovery_source_frames']
                assert e['frame']-e['confirmation_start']+1 == e['confirmation_frames']
                assert e['identity_status'] == 'requires_identity_review'
                # No retroactive coordinates were inserted before confirmation.
                assert not any(r['player_slot'] == e['player_slot']
                               and e['last_observed_frame'] < int(r['source_frame']) < e['frame'] for r in rows)
                report['reacquisitions'].append(dict(tag=tag,**e))
        old = baseline[tag]
        diff_frames = set()
        if old.get('tracking_summary'):
            oldrows = a.read(OLD/'results'/f'players_{tag}_frozen_stage2'/'selected_four.csv')
            oldframes = grouped(oldrows)
            oldseeds = old['tracking_summary']['seeds']
            init = next(e for e in events if e['event'] == 'initialized')
            assert init['seed_ids'] == oldseeds,'Cross-version slot labels cannot be compared'
            for frame,rs in byframe.items():
                for new in rs:
                    same = [r for r in oldframes[frame] if r['player_slot'] == new['player_slot']]
                    own_iou = max((a.box_iou(new,r) for r in same),default=0.)
                    other = [r for r in oldframes[frame] if r['player_slot'] != new['player_slot']
                             and r['player_slot'].split('_')[0] == new['player_slot'].split('_')[0]]
                    for r in other:
                        other_iou = a.box_iou(new,r)
                        if other_iou >= .5 and own_iou < .3:
                            differences.append(dict(tag=tag,source_frame=frame,new_slot=new['player_slot'],
                                old_slot_on_same_box=r['player_slot'],new_raw_id=new['track_id'],
                                old_raw_id=r['track_id'],iou=other_iou,
                                status='cross_version_disagreement_not_proven_identity_switch'))
                            diff_frames.add(frame)
        before = old.get('tracking_summary',{}).get('counts',{}).get('four_simultaneous')
        report['comparison'].append(dict(tag=tag,total_frames=summary['frames'],
            previous_four_assigned=before,current_four_assigned=summary['counts']['four_simultaneous'],
            initialization_wait_frames=summary['counts']['pre_initialization_frames'],
            reacquisition_proposals=summary['reacquisition_proposals'],
            cross_version_disagreement_frames=len(diff_frames)))
        video = folder/'tracking_review_h264.mp4'
        meta = json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error',
            '-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames',
            '-of','json',str(video)]))['streams'][0]
        assert int(meta['nb_frames']) == summary['frames']
        assert meta['codec_name'] == 'h264' and meta['pix_fmt'] == 'yuv420p'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],
                       capture_output=True,check=True)
        report['media'].append(dict(tag=tag,full_decode='passed',**meta))
    if differences:
        with (OUT/'cross_version_disagreements.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(differences[0]))
            w.writeheader()
            w.writerows(differences)
    (OUT/'integrity_and_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['comparison'],indent=2))
    print('Verified: input boxes preserved; one-to-one assignments; no early/backfilled observations; confirmation timing; media decode; source hashes.')


if __name__ == '__main__':
    main()
