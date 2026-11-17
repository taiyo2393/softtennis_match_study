"""Create a compact review queue from selected-player gaps and ID links."""
import argparse
import csv
import json
from pathlib import Path
import cv2

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=('bytetrack','botsort_reid'),default='bytetrack')
    args=p.parse_args()
    items=[]
    for pilot in ('pilot_003','pilot_004','pilot_005'):
        directory=args.root/pilot
        summary=next(x for x in json.loads((directory/'four_player_selection_summary.json').read_text(encoding='utf-8')) if x['mode']==args.mode)
        first=summary['seed_frame']
        for slot,detail in summary['slots'].items():
            for start,end in detail['missing_runs']:
                if end-start>=30:
                    priority='high'
                else:
                    priority='medium'
                items.append(dict(pilot=pilot,slot=slot,start_frame=start,end_frame=end,
                                  reason='player_missing',priority=priority))
        for link in summary['stitches']:
            f=link['start_frame']
            items.append(dict(pilot=pilot,slot=link['slot'],start_frame=f-1,end_frame=f+1,
                              reason=link['link_type'],priority='high' if 'review' in link['link_type'] else 'medium'))
        # Nearby gap and ID-link events belong to one review decision.
        own=sorted([i for i in items if i['pilot']==pilot],key=lambda x:(x['slot'],x['start_frame']))
        merged=[]
        for item in own:
            if merged and merged[-1]['slot']==item['slot'] and item['start_frame']<=merged[-1]['end_frame']+8:
                prev=merged[-1]
                prev['end_frame']=max(prev['end_frame'],item['end_frame'])
                prev['reason']+='+'+item['reason']
                if item['priority']=='high': prev['priority']='high'
            else:
                merged.append(item)
        items=[i for i in items if i['pilot']!=pilot]+merged
        video=cv2.VideoCapture(str(directory/f'{args.mode}_selected_four.mp4'))
        fps=video.get(cv2.CAP_PROP_FPS)
        length=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for index,item in enumerate(merged,1):
            begin=max(0,item['start_frame']-first-20)
            finish=min(length-1,item['end_frame']-first+20)
            item['source_start_sec']=round(item['start_frame']/fps,3)
            item['source_end_sec']=round(item['end_frame']/fps,3)
            output=directory/f'{args.mode}_review_{index:02d}.mp4'
            item['preview_path']=str(output.resolve())
            video.set(cv2.CAP_PROP_POS_FRAMES,begin)
            writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720))
            if not writer.isOpened(): raise RuntimeError(f'Cannot write {output}')
            for _ in range(begin,finish+1):
                ok,frame=video.read()
                if not ok: break
                writer.write(frame)
            writer.release()
        video.release()
    output=args.root/f'{args.mode}_review_queue.csv'
    with output.open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(items[0]))
        writer.writeheader();writer.writerows(items)
    print(json.dumps({'review_items':len(items),'queue':str(output),'items':items},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
