import csv,json,statistics
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
with (p/'exploratory_position_outcome_features.csv').open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
rs=[r for r in rs if r['position_count']=='4'];paired=[];rows=[]
for r in rs:
 server=r['server_side_candidate'];recv='far' if server=='near' else 'near';win=r['winner_side']==server
 for role,side in [('server',server),('receiver',recv)]:rows.append(dict(candidate_id=r['candidate_id'],pair_role=role,pair_won=(win if role=='server' else not win),lateral_gap=float(r[side+'_lateral_gap']),longitudinal_gap=float(r[side+'_longitudinal_gap']),server_label_status=r['server_review'],unit='one_pair_within_rally_not_independent_from_opposing_pair'))
 for axis in ['lateral','longitudinal']:paired.append(dict(candidate_id=r['candidate_id'],axis=axis,server_minus_receiver_gap=float(r[server+'_'+axis+'_gap'])-float(r[recv+'_'+axis+'_gap']),server_won=win))
summary=[]
for role in ['server','receiver']:
 for won in [True,False]:
  subset=[r for r in rows if r['pair_role']==role and r['pair_won']==won]
  summary.append(dict(role=role,won=won,n=len(subset),candidates=[r['candidate_id'] for r in subset],lateral_median=statistics.median(r['lateral_gap'] for r in subset),longitudinal_median=statistics.median(r['longitudinal_gap'] for r in subset)))
with (p/'serve_role_position_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(p/'serve_role_position_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
fig,axs=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
for ax,axis in zip(axs,['lateral','longitudinal']):
 for j,r in enumerate(rs):
  s=r['server_side_candidate'];o='far' if s=='near' else 'near';win=r['winner_side']==s;x1=100*float(r[s+'_'+axis+'_gap']);x2=100*float(r[o+'_'+axis+'_gap']);ax.plot([x1,x2],[j,j],color='#bbb',lw=1);ax.scatter(x1,j,color='#206a99',marker='o',label='Serving pair' if j==0 else None);ax.scatter(x2,j,color='#bd681a',marker='s',label='Receiving pair' if j==0 else None)
 ax.set_yticks(range(len(rs)),[r['candidate_id']+(' S win' if r['winner_side']==r['server_side_candidate'] else ' R win') for r in rs]);ax.set_title(axis.capitalize()+' gap');ax.set_xlabel('% of court '+('width' if axis=='lateral' else 'length'));ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Exploratory role separation: serving winners 7, receiving winners 2');fig.tight_layout();fig.savefig(p/'serve_role_position_comparison.png',dpi=180);plt.close(fig)
