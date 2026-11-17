"""Run integrity checks and summarize the development comparison."""
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import improve_players as m

OUT = Path(__file__).resolve().parent


def check_rules():
    row = dict(track_id=1,x1=1000,y1=1200,x2=1200,y2=1500,bottom_center_x=1100,bottom_center_y=1500)
    negative = np.array([[.2,.8,0],[.2,.8,0]],dtype=np.float32)
    state = dict(row=row,frame=10,velocity=np.zeros(2),side='near',reference_height=300,
                 template=np.array([[.8,.2,0],[.8,.2,0]],dtype=np.float32),negatives=[negative])
    assert m.original_cost(state,row,negative,11) > 1e5
    assert m.continuity_cost(state,row,negative,11) < 1
    assert m.continuity_cost(state,dict(row,track_id=2),negative,11) > 1e5
    assert m.continuity_cost(state,row,negative,13) > 1e5
    fragment = dict(row,y1=1400)
    assert m.continuity_cost(state,fragment,negative,11) > 1e5
    print('Rule checks passed: continuous identity exception, new-ID/long-gap veto, fragment rejection.')


def main():
    check_rules()
    report = dict(scope='Development on 044/056, no held-out identity accuracy',comparison=[],media=[])
    for tag in ('044','056'):
        variants = [('previous',OUT.parent/'tracking_quality_20261102'/f'players_{tag}_appearance_motion_guarded'),
                    ('continuity_size_only',OUT/f'players_{tag}_continuity_size_guard'),
                    ('continuity_size_and_crop',OUT/f'players_{tag}_continuity_size_guard_far_crop')]
        original = m.assoc.read(m.assoc.BASE/f'full_rally_{tag}'/'bytetrack.csv')
        config = json.loads((m.assoc.BASE/f'full_rally_{tag}'/'four_player_selection_summary.json').read_text(encoding='utf-8'))[0]
        original = [r for r in original if config['seed_frame'] <= int(r['source_frame']) <= config['analysis_ready_end_frame']]
        fused = m.assoc.read(OUT/f'augmented_detections_{tag}.csv')
        restored = [r for r in fused if r['candidate_origin'] == 'original_full_frame']
        # Crop addition must not change or delete existing detector rows.
        assert len(restored) == len(original)
        for old,new in zip(original,restored):
            assert all(str(new[k]) == str(v) for k,v in old.items())
        for name,folder in variants:
            summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            rows = m.assoc.read(folder/'selected_four.csv')
            byframe = defaultdict(list)
            seed_heights = {slot:2*(box[3]-box[1]) for slot,(_,_,box) in
                            zip(m.assoc.SLOTS,summary['initial_seed_boxes'])}
            reference_rows = {r['player_slot']:r for r in rows if int(r['source_frame']) == summary['first_frame']}
            for r in rows:
                assert r['position_source'] == 'detected_box'
                assert r['identity_status'] == 'candidate_not_ground_truth'
                assert summary['first_frame'] <= int(r['source_frame']) <= summary['last_frame']
                if name != 'previous':
                    assert m.assoc.height(r) >= .50*m.assoc.height(reference_rows[r['player_slot']])-1e-5
                byframe[int(r['source_frame'])].append(r)
            for rs in byframe.values():
                assert len({r['player_slot'] for r in rs}) == len(rs) <= 4
                assert len({r['track_id'] for r in rs}) == len(rs)
            report['comparison'].append(dict(rally=tag,variant=name,frames=summary['frames'],
                                              counts=summary['counts'],
                                              selected_crop_observations=sum(r.get('candidate_origin') == 'far_crop_extra' for r in rows)))
        video = variants[-1][1]/'tracking_review_h264.mp4'
        meta = json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error',
                    '-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames',
                    '-of','json',str(video)]))['streams'][0]
        assert int(meta['nb_frames']) == (1035 if tag == '044' else 770)
        assert meta['codec_name'] == 'h264' and meta['pix_fmt'] == 'yuv420p'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],
                       check=True,capture_output=True)
        report['media'].append(dict(path=str(video),full_decode='passed',**meta))
    (OUT/'comparison_and_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['comparison'],indent=2))
    print('Original rows preserved; assignments and fragments checked; both preview videos fully decoded.')


if __name__ == '__main__':
    main()
