"""Validate frozen-run integrity and summarize missing runs, without inferring GT."""
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    protocol = json.loads((OUT/'predeclared_protocol.json').read_text(encoding='utf-8'))
    results = json.loads((OUT/'validation_results.json').read_text(encoding='utf-8'))
    assert len(results) == len(protocol['intervals']) == 3
    for path,digest in protocol['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,path
    audit = dict(settings_hashes_unchanged=True,initialization_attempts=3,initialization_proposals=0,
                 initialization_withheld=0,missing_runs=[],media=[],warning='Missing assignments are not proven tracking errors. No independent identity accuracy measured.')
    gaps = []
    for r in results:
        tag = r['tag']
        if r['initialization_status'] != 'four_seeds_proposed':
            audit['initialization_withheld'] += 1
            continue
        audit['initialization_proposals'] += 1
        folder = OUT/'results'/f'players_{tag}_frozen_stage2'
        rows = read(folder/'selected_four.csv')
        raw = read(OUT/'crop_runs'/f'augmented_detections_{tag}.csv')
        original = read(OUT/'detector_runs'/f'full_rally_{tag}'/'bytetrack.csv')
        restored = [{k:v for k,v in row.items() if k != 'candidate_origin'} for row in raw
                    if row['candidate_origin'] == 'original_full_frame']
        assert restored == original
        byframe = defaultdict(list)
        for row in rows:
            byframe[int(row['source_frame'])].append(row)
            assert row['position_source'] == 'detected_box'
            assert row['identity_status'] == 'candidate_not_ground_truth'
        for rs in byframe.values():
            assert len({x['player_slot'] for x in rs}) == len(rs) <= 4
            assert len({x['track_id'] for x in rs}) == len(rs)
        simultaneous = sum(len(rs) == 4 for rs in byframe.values())
        assert simultaneous == r['tracking_summary']['counts']['four_simultaneous']
        audit.setdefault('availability',[]).append(dict(tag=tag,frames=r['frames'],four_assigned=simultaneous,
                                                         fraction=simultaneous/r['frames']))
        for slot in ('near_front','near_rear','far_rear','far_front'):
            selected = {int(x['source_frame']):x for x in rows if x['player_slot'] == slot}
            runs = []
            for f in range(r['first_frame'],r['last_frame']+1):
                if f in selected:
                    continue
                if runs and runs[-1][1] == f-1:
                    runs[-1][1] = f
                else:
                    runs.append([f,f])
            for a,b in runs:
                last = selected.get(a-1)
                tid = last['track_id'] if last else ''
                gap = dict(tag=tag,player_slot=slot,start_frame=a,end_frame=b,frames=b-a+1,
                           duration_sec=(b-a+1)/r['source_fps'],last_raw_id=tid,
                           same_raw_id_present_frames=sum(x['track_id'] == tid and a <= int(x['source_frame']) <= b for x in raw),
                           exceeds_90_frame_reactivation_limit=b-a+1>90,
                           interpretation='assignment_gap_not_ground_truth_error')
                gaps.append(gap)
        video = folder/'tracking_review_h264.mp4'
        meta = json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error',
            '-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames',
            '-of','json',str(video)]))['streams'][0]
        assert int(meta['nb_frames']) == r['frames']
        assert meta['codec_name'] == 'h264' and meta['pix_fmt'] == 'yuv420p'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],
                       capture_output=True,check=True)
        audit['media'].append(dict(path=str(video),full_decode='passed',**meta))
    with (OUT/'missing_run_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(gaps[0]))
        writer.writeheader()
        writer.writerows(gaps)
    audit['missing_runs'] = gaps
    audit['long_missing_runs'] = [g for g in gaps if g['frames'] >= 30]
    (OUT/'validation_integrity_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
