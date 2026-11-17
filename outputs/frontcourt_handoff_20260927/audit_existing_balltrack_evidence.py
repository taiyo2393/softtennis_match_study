import csv,json,math,statistics
from pathlib import Path
from collections import Counter
root=Path(r'C:\Users\ta239\Research\softtennis_match_study\study');p=Path(__file__).resolve().parent
with (root/'annotations/balltrack_groundtruth_match1_pilot.csv').open(encoding='utf-8-sig',newline='') as f:labels=list(csv.DictReader(f))
cache={};continuity=[]
for cid in sorted({r['candidate_id'] for r in labels}):
 folder=root/'data/balltrack/racketvision/match1'/cid/'runs/threshold030_median120'
 with (folder/'balltrack_predictions.csv').open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
 cache[cid]={int(r['local_frame']):r for r in rs};meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'));gaps=[];gap=0
 for r in rs:
  if int(r['visible'])==0:gap+=1
  elif gap:gaps.append(gap);gap=0
 if gap:gaps.append(gap)
 continuity.append(dict(candidate_id=cid,frames=len(rs),candidate_present=sum(int(r['visible']) for r in rs),output_fraction=sum(int(r['visible']) for r in rs)/len(rs),max_no_output_gap_frames=max(gaps,default=0),max_no_output_gap_sec=max(gaps,default=0)/meta['fps'],source_start_sec=meta['clip_start_timestamp_sec'],scope='candidate_continuity_not_correct_ball_tracking'))
counts=Counter();errors=[]
for r in labels:
 if r['evaluation_split']!='test':continue
 state=r['truth_visibility'];counts['total']+=1
 if state not in ['visible','not_visible']:counts[state]+=1;continue
 pred=cache[r['candidate_id']][int(r['local_frame'])];vis=int(pred['visible'])==1;truth=state=='visible';counts['evaluable']+=1;counts['truth_visible']+=truth;counts['pred_visible']+=vis
 if truth and vis:
  e=math.hypot(float(pred['x_px'])-float(r['truth_x_px']),float(pred['y_px'])-float(r['truth_y_px']));errors.append(e)
  counts['presence_tp']+=1
  if e<=16:counts['localized_tp']+=1
  else:counts['wrong_localization']+=1
 elif truth:counts['presence_fn']+=1
 elif vis:counts['presence_fp']+=1
 else:counts['presence_tn']+=1
original=json.loads((root/'analysis/balltrack_evaluation/match1_test_threshold030_median120_summary.json').read_text(encoding='utf-8'))
for k in ['presence_tp','presence_fn','presence_fp','presence_tn','localized_tp','wrong_localization']:assert counts[k]==original['counts'][k],k
result=dict(recomputed_test_counts=dict(counts),localization_recall=counts['localized_tp']/counts['truth_visible'],image_size='1920x1080 clips; 16px tolerance',mean_distance_when_both_visible=statistics.mean(errors),annotation_rows=len(labels),reviewer_id_nonempty=sum(bool(r['reviewer_id']) for r in labels),label_confidence_nonempty=sum(bool(r['label_confidence']) for r in labels),warning='Label provenance not certified by nonempty visibility values; no direction accuracy measured')
(p/'balltrack_existing_evidence_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (p/'balltrack_existing_output_continuity.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=continuity[0]);w.writeheader();w.writerows(continuity)
print(json.dumps(result));print('Output continuity examples:',json.dumps(continuity[:3]));print('20 clips audited; existing selected setting retained; source outputs unchanged')
