import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from adaptive_near import old
from run_adaptive_near import baseline,write

ROOT=Path(__file__).resolve().parent


def main():
    reports=[]
    for tag in ('003','002','005','007','048'):
        folder=ROOT/f'players_{tag}_temporal'
        protocol=json.loads((ROOT/f'protocol_{tag}.json').read_text(encoding='utf-8'))
        for path,digest in protocol['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        comparison=json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
        farpath=(ROOT/'recompute/tracking_quality_20261102_stage7/players_003_global/far_pair_path.csv') if tag=='003' else folder/'frozen_far_path.csv'
        near=old.a.read(folder/'near_pair_path.csv');far=old.a.read(farpath)
        assert (folder/'frozen_far_path.csv').read_bytes()==Path(protocol['far_path_source']).read_bytes()
        raw={(r['source_frame'],r['track_id']):r for r in old.a.read(Path(summary['input_csv']))}
        combined=[];fields=[];byframe=defaultdict(list)
        for branch,rows in [('adaptive_near',near),('recomputed_global_far' if tag=='003' else 'preserved_global_far',far)]:
            for r in rows:
                assert r['position_source']=='detected_box'
                source=raw[(r['source_frame'],r['track_id'])]
                assert all(r[k]==source[k] for k in ('x1','y1','x2','y2','bottom_center_x','bottom_center_y'))
                row=dict(r,processing_branch=branch)
                for k in row:
                    if k not in fields:fields.append(k)
                combined.append(row);byframe[int(r['source_frame'])].append(row)
        for rs in byframe.values():
            assert len({r['player_slot'] for r in rs})==len(rs)<=4
            assert len({r['track_id'] for r in rs})==len(rs)
        combined=[{k:r.get(k,'') for k in fields} for r in combined]
        combined.sort(key=lambda r:(int(r['source_frame']),r['player_slot']))
        write(folder/'four_player_path.csv',combined)
        if tag!='003':
            assert comparison['removed_near_rows']==comparison['added_near_rows']==0
        video=(ROOT/'recompute/tracking_quality_20261102_stage7/players_003_global/global_review_h264.mp4') if tag=='003' else folder/'combined_review_h264.mp4'
        media=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames','-of','json',str(video)]))['streams'][0]
        assert int(media['nb_frames'])==summary['frames'] and media['codec_name']=='h264' and media['pix_fmt']=='yuv420p'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],capture_output=True,check=True)
        reports.append(dict(comparison,final_far_path_source=str(farpath),
            final_far_path_matches_previous=farpath.read_bytes()==Path(protocol['far_path_source']).read_bytes(),
            final_video=str(video),combined_four_observed_frames=sum(len(rs)==4 for rs in byframe.values()),
            observed_positions_match_source_boxes=True,one_to_one_assignments=True,media=dict(full_decode='passed',**media)))
    (ROOT/'integrity_and_regression_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
