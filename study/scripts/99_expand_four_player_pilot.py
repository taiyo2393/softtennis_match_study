"""Run a bounded ByteTrack test across existing, human-labeled match1 events."""

import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts'
OUT=ROOT/'analysis/person_tracker_comparison'
EVENTS=[
    ('pilot_007','match1_direction_match1_remaining_01_062',36464),
    ('pilot_008','match1_direction_pilot_01_011',62553),
    ('pilot_009','match1_direction_match1_remaining_01_106',72024),
    ('pilot_010','match1_far_front_direction_02_046',94383),
    ('pilot_011','match1_far_front_direction_02_059',115440),
    ('pilot_012','match1_far_front_direction_02_078',140961),
    ('pilot_013','match1_direction_match1_remaining_01_190',154156),
]

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))

def run(command):
    return subprocess.run(command,capture_output=True,text=True,check=False)

def main():
    labels={r['event_id']:r for r in read(ROOT/'annotations/match1_target_front_identity_review.csv')}
    intervals=read(ROOT/'data/segments/analysis_ready_main_camera_segments.csv')
    records=[]
    for pilot,event,impact in EVENTS:
        label=labels[event]
        assert int(label['impact_source_frame'])==impact and label['identity_status']=='confirmed'
        start=impact-60
        assert any(int(s['start_source_frame'])<=start and int(s['end_source_frame'])>=impact+119
                   and s['usable_for_analysis'].lower()=='true' for s in intervals)
        directory=OUT/pilot
        row=dict(pilot=pilot,event_id=event,impact_frame=impact,
                 target_camera_end=label['target_front_camera_end'],
                 target_player_id=label['target_front_player_id'],status='',
                 observed_player_frames='',possible_player_frames='',
                 missing_player_frames='',id_links='',note='')
        if directory.exists():
            row['status']='already_exists'
            records.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
            continue
        tracked=run([sys.executable,str(SCRIPTS/'94_compare_person_trackers.py'),
                     '--start-frame',str(start),'--frames','180','--mode','bytetrack',
                     '--output',str(directory)])
        if tracked.returncode:
            row['status']='tracking_failed';row['note']=tracked.stderr[-800:]
        else:
            selected=run([sys.executable,str(SCRIPTS/'96_select_four_players_pilot.py'),
                          '--pilot',str(directory),'--mode','bytetrack'])
            if selected.returncode:
                row['status']='auto_seed_unresolved';row['note']=selected.stderr[-500:]
            else:
                summary=json.loads((directory/'four_player_selection_summary.json').read_text(encoding='utf-8'))[0]
                observed=sum(s['observed_frames'] for s in summary['slots'].values())
                possible=4*next(iter(summary['slots'].values()))['total_frames']
                row.update(status='selected',observed_player_frames=observed,
                           possible_player_frames=possible,missing_player_frames=possible-observed,
                           id_links=len(summary['stitches']))
                rendered=run([sys.executable,str(SCRIPTS/'97_render_four_player_pilot.py'),
                              '--pilot',str(directory),'--mode','bytetrack'])
                if rendered.returncode:
                    row['status']='render_failed';row['note']=rendered.stderr[-500:]
                else:
                    encoded=run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',
                                 str(directory/'bytetrack_selected_four.mp4'),'-c:v','libx264',
                                 '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags',
                                 '+faststart',str(directory/'bytetrack_selected_four_h264.mp4')])
                    if encoded.returncode:
                        row['status']='h264_failed';row['note']=encoded.stderr[-500:]
        records.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    output=OUT/'expanded_pilot_summary.csv'
    with output.open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    print(f'Wrote {output}',flush=True)

if __name__=='__main__':
    main()
