import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from identity_guard import old

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'conservative'


def main():
    protocol = json.loads((OUT/'experiment_protocol.json').read_text(encoding='utf-8'))
    for path,digest in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    report = dict(scope='Abstention/integrity checks only; identity accuracy remains unmeasured',runs=[])
    results = json.loads((OUT/'experiment_results.json').read_text(encoding='utf-8'))
    for s in results:
        folder = OUT/f'players_{s["tag"]}_temporal'
        labeled = old.a.read(folder/'selected_four.csv')
        unknown_path = folder/'unresolved_far_detections.csv'
        unknown = old.a.read(unknown_path) if unknown_path.exists() else []
        raw = old.a.read(Path(s['input_csv']))
        rawkey = {(r['source_frame'],r['track_id']):r for r in raw}
        last,byframe = {},defaultdict(list)
        for r in sorted(labeled,key=lambda r:(int(r['source_frame']),r['player_slot'])):
            slot,frame = r['player_slot'],int(r['source_frame'])
            assert r['identity_status'] == 'candidate_not_ground_truth'
            assert r['position_source'] == 'detected_box'
            if slot in last and s['pair_policy']['initial_pair_color_similarity'][slot.split('_')[0]]['not_distinct_by_color']:
                assert frame-last[slot] <= 2,'Similar-clothes identity restored across a long gap'
            last[slot] = frame
            byframe[frame].append(r)
            original = rawkey[(r['source_frame'],r['track_id'])]
            assert all(r[k] == original[k] for k in ('x1','y1','x2','y2','bottom_center_x','bottom_center_y'))
        labeled_keys = {(r['source_frame'],r['track_id']) for r in labeled}
        for r in unknown:
            assert r['player_slot'] == '' and r['identity_status'] == 'unresolved'
            assert (r['source_frame'],r['track_id']) not in labeled_keys
            original = rawkey[(r['source_frame'],r['track_id'])]
            assert all(r[k] == original[k] for k in ('x1','y1','x2','y2','bottom_center_x','bottom_center_y'))
        for rows in byframe.values():
            assert len({r['player_slot'] for r in rows}) == len(rows) <= 4
            assert len({r['track_id'] for r in rows}) == len(rows)
        events = json.loads((folder/'events.json').read_text(encoding='utf-8'))
        for event in events:
            if event['event'] == 'reacquired_candidate':
                side = event['player_slot'].split('_')[0]
                assert not s['pair_policy']['initial_pair_color_similarity'][side]['not_distinct_by_color']
        video = folder/'identity_guard_review_h264.mp4'
        meta = json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error',
            '-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames',
            '-of','json',str(video)]))['streams'][0]
        assert int(meta['nb_frames']) == s['frames']
        assert meta['codec_name'] == 'h264' and meta['pix_fmt'] == 'yuv420p'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],
                       capture_output=True,check=True)
        report['runs'].append(dict(tag=s['tag'],labeled_rows=len(labeled),unknown_rows=len(unknown),
            four_labeled_frames=s['counts']['four_simultaneous'],
            withheld_handover_frames=s['identity_guard']['withheld_handover_frames'],
            full_decode='passed',**meta))
    (OUT/'integrity_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
