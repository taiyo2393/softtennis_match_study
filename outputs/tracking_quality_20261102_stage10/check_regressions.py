import json
import subprocess
from pathlib import Path
from collections import defaultdict
import gap_pair_tracking as g

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent


def old_far(tag):
    if tag=='003':return BASE/'tracking_quality_20261102_stage9/recompute/tracking_quality_20261102_stage7/players_003_global/far_pair_path.csv'
    if tag in ('007','025'):return BASE/'tracking_quality_20261102_stage8/tracking_quality_20261102_stage7'/f'players_{tag}_global/far_pair_path.csv'
    return BASE/'tracking_quality_20261102_stage7'/f'players_{tag}_global/far_pair_path.csv'


def main():
    reports=[];fields=('source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y')
    g.OUT=ROOT/'experiment/tracking_quality_20261102_stage7'
    for tag in ('025','002','003','005','007','048'):
        folder=g.OUT/f'players_{tag}_global'
        summary,event,near,cache=g.candidates(tag)
        current=g.a.read(folder/'far_pair_path.csv');prior=g.a.read(old_far(tag))
        before={tuple(r[k] for k in fields) for r in prior};after={tuple(r[k] for k in fields) for r in current}
        if tag!='025':assert before==after,tag
        raw={(r['source_frame'],r['track_id']):r for r in g.a.read(Path(summary['input_csv']))}
        frame_rows=defaultdict(list)
        for r in current:
            original=raw[(r['source_frame'],r['track_id'])]
            assert all(r[k]==original[k] for k in fields if k not in ('player_slot',))
            frame_rows[int(r['source_frame'])].append(r)
        for f,rs in frame_rows.items():
            allrows=rs+near[f]
            assert len({r['track_id'] for r in allrows})==len(allrows)<=4
            assert len({r['player_slot'] for r in allrows})==len(allrows)
        video=folder/'global_review_h264.mp4'
        media=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames','-of','json',str(video)]))['streams'][0]
        assert int(media['nb_frames'])==summary['frames'] and media['codec_name']=='h264' and media['pix_fmt']=='yuv420p'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],capture_output=True,check=True)
        reports.append(dict(tag=tag,previous_path_exactly_preserved=before==after,
            old_rows=len(prior),new_rows=len(current),shared_rows=len(before&after),
            input_boxes_preserved=True,one_to_one_across_both_sides=True,media=dict(full_decode='passed',**media)))
    (ROOT/'regression_and_integrity_checks.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
