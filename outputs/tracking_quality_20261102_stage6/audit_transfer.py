"""Check frozen-parameter transfer, original boxes, directional consistency and media."""
import hashlib
import json
import subprocess
from pathlib import Path
import global_pair_tracking as g

ROOT = Path(__file__).resolve().parent


def main():
    baseline = json.loads((ROOT/'players_048_global/protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'global_pair_tracking.py').read_bytes()).hexdigest() == baseline['code_sha256']
    reports = []
    for tag in ('002','005'):
        folder = ROOT/f'players_{tag}_global'
        protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
        assert protocol == baseline,'Parameters or implementation differ from the reviewed 048 run'
        summary,event,near,cache = g.candidates(tag)
        rows = g.a.read(folder/'far_pair_path.csv')
        keyed = {}
        for r in rows:
            frame = int(r['source_frame'])
            assert event['frame'] <= frame <= summary['last_frame']
            assert r['position_source'] == 'detected_box'
            matches = [c['row'] for c in cache[frame-event['frame']] if str(c['row']['track_id']) == r['track_id']]
            assert len(matches)==1
            assert all(str(matches[0][k])==r[k] for k in ('x1','y1','x2','y2','bottom_center_x','bottom_center_y'))
            keyed.setdefault(frame,{})[r['player_slot']] = r
        for rs in keyed.values():
            assert len(rs)<=2 and len({r['track_id'] for r in rs.values()})==len(rs)
        last = max(f for f,rs in keyed.items() if len(rs)==2)
        ids = {s:int(r['track_id']) for s,r in keyed[last].items()}
        rev = list(reversed(cache[:last-event['frame']+1]))
        path,_ = g.solve(rev,dict(seed_ids=ids))
        start_ids,comparisons = {},[]
        for offset,choices in enumerate(path):
            frame = last-offset
            for i,index in enumerate(choices):
                if index<0:continue
                slot=('far_rear','far_front')[i]
                row=rev[offset][index]['row']
                if frame==event['frame']:start_ids[slot]=int(row['track_id'])
                forward=keyed.get(frame,{}).get(slot)
                if forward:comparisons.append(dict(source_frame=frame,slot=slot,iou=g.a.box_iou(row,forward)))
        g.write(folder/'forward_backward_comparison.csv',comparisons)
        video=folder/'global_review_h264.mp4'
        meta=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames','-of','json',str(video)]))['streams'][0]
        assert int(meta['nb_frames'])==summary['frames'] and meta['codec_name']=='h264' and meta['pix_fmt']=='yuv420p'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(video),'-f','null','-'],capture_output=True,check=True)
        report=dict(tag=tag,parameters_and_code_match_048=True,original_boxes_preserved=True,
            initial_ids={s:event['seed_ids'][s] for s in ('far_rear','far_front')},
            reverse_anchor_frame=last,reverse_anchor_ids=ids,reverse_returned_initial_ids=start_ids,
            jointly_observed_comparisons=len(comparisons),agreeing_boxes_iou_ge_half=sum(r['iou']>=.5 for r in comparisons),
            note='Reverse anchors derived from forward proposal; consistency is not independent identity accuracy.',
            human_review_status='not_reviewed',media=dict(full_decode='passed',**meta))
        (folder/'directional_and_integrity_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        reports.append(report)
        print(json.dumps(report),flush=True)
    (ROOT/'transfer_audit_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')


if __name__=='__main__':main()
