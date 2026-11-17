import csv,json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent

def read(n):
 with (p/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
labels={r['candidate_id'][-3:]:r for r in read('rally_label_review.csv') if r['winner_camera_side_review_status']=='human_confirmed'};points=[]
def add(n,frame,side,x,y,source,px,py):points.append(dict(candidate_id=n,source_frame=frame,camera_side=side,x_normalized=x,y_normalized=y,image_x_px=px,image_y_px=py,source_file=source,position_basis='AI_bbox_bottom_approximation',winner_side_human=labels[n]['winner_camera_side'],winner_team_derived=labels[n]['winner_team_id'],sampling_status='exploratory_contact_anchor_uncertainty_retained'))
for name in ['additional_four_player_snapshot_proposals.csv','six_service_selected_position_candidates.csv','048_selected_position_candidates.csv']:
 for r in read(name):
  if r.get('court_x_candidate','')=='':continue
  add(r['candidate_id'],r.get('sample_frame') or r.get('source_frame'),r['image_slot'].split('_')[0],float(r['court_x_candidate']),float(r['court_y_candidate']),name,r['bottom_x_px'],r['bottom_y_px'])
for r in read('044_second_snapshot_candidates.csv'):add('044',r['sample_source_frame'],r['player_slot'].split('_')[0],float(r['court_x_normalized']),float(r['court_y_normalized']),'044_second_snapshot_candidates.csv',r['bottom_center_x_px'],r['bottom_center_y_px'])
for r in read('serve_phase_position_snapshot_candidates.csv'):
 if r['candidate_id'].endswith('_056'):add('056',r['sample_source_frame'],r['player_slot'].split('_')[0],float(r['court_x_normalized']),float(r['court_y_normalized']),'serve_phase_position_snapshot_candidates.csv',r['bottom_center_x_px'],r['bottom_center_y_px'])
assert len(points)==42
features=[]
for n in sorted(labels):
 ps=[r for r in points if r['candidate_id']==n];row=dict(candidate_id=n,source_frame=ps[0]['source_frame'],winner_side=labels[n]['winner_camera_side'],winner_team=labels[n]['winner_team_id'],position_count=len(ps),status='four_position_candidates' if len(ps)==4 else 'partial_occlusion',server_side_candidate='far' if n in ['044','048'] else 'near',server_review='AI_sequence_inferred_not_independent_human_label')
 for side in ['near','far']:
  pair=[r for r in ps if r['camera_side']==side];row[side+'_lateral_gap']='';row[side+'_longitudinal_gap']=''
  if len(pair)==2:
   row[side+'_lateral_gap']=abs(pair[0]['x_normalized']-pair[1]['x_normalized']);row[side+'_longitudinal_gap']=abs(pair[0]['y_normalized']-pair[1]['y_normalized'])
 for axis in ['lateral','longitudinal']:
  win=row['winner_side'];lose='far' if win=='near' else 'near';a,b=row[win+'_'+axis+'_gap'],row[lose+'_'+axis+'_gap'];row['winner_minus_loser_'+axis+'_gap']=a-b if a!='' and b!='' else ''
 features.append(row)
for name,rows in [('exploratory_position_outcome_points.csv',points),('exploratory_position_outcome_features.csv',features)]:
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
complete=[r for r in features if r['position_count']==4];stats={}
for axis in ['lateral','longitudinal']:
 d=[r['winner_minus_loser_'+axis+'_gap'] for r in complete];stats[axis]=dict(n=len(d),winner_larger=sum(x>0 for x in d),winner_smaller=sum(x<0 for x in d),median_winner_minus_loser=statistics.median(d),range=[min(d),max(d)])
(p/'exploratory_position_outcome_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');print(json.dumps(stats));print('Complete candidates:',[r['candidate_id'] for r in complete])
try:
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
 for ax,axis,title in zip(axes,['lateral','longitudinal'],['Lateral gap (fraction of court width)','Longitudinal gap (fraction of court length)']):
  d=[100*r['winner_minus_loser_'+axis+'_gap'] for r in complete];ax.axvline(0,color='gray',lw=1);ax.scatter(d,range(len(d)),color='#176f9a');ax.set_yticks(range(len(d)),[r['candidate_id'] for r in complete]);ax.set_title(title,fontsize=10);ax.set_xlabel('Winner minus loser (percentage points)');ax.grid(axis='x',alpha=.2)
 axes[0].set_ylabel('Candidate rally');fig.suptitle('Exploratory only: 9 snapshots from one match',fontsize=12);fig.tight_layout();fig.savefig(p/'exploratory_pair_gap_comparison.png',dpi=180);plt.close(fig)
except ImportError:print('Plot library unavailable; CSV and summary saved')
