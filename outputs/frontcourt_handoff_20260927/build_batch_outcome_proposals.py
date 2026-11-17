import csv
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parent

def read(n):
 with (p/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,rows,fields=None):
 with (p/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
labels=read('rally_label_review.csv');scores=read('score_change_suggestions.csv');frames={r['candidate_id']:r for r in read('scoreboard_scan_queue.csv')};by={r['candidate_id']:r for r in scores}
r=next(r for r in labels if r['candidate_id']=='match1_rally_window_007');r.update(winner_camera_side='far',winner_camera_side_review_status='human_confirmed',winner_team_id='上岡俊介・丸山海斗',team_side_mapping_status='initial_anchor_first_game_change_and_ai_checkpoint_consistent',winner_evidence_source_frame=frames[r['candidate_id']]['end_plus_1s_source_frame'],winner_evidence_type='human_video_side_review_and_scoreboard_bottom_2_to_3',winner_review_status='human_side_plus_score_team_derived',reviewer='human_user side; Codex scoreboard linkage 2026-11-02')
note='User answered far for 007_score_point_context_h264.mp4. Team derived from first-game court change and bottom 2 to 3; exact boundaries and positions unconfirmed.'
if note not in r['review_note']:r['review_note']=(r['review_note']+' | '+note).strip(' |')
write('rally_label_review.csv',labels)
log=read('human_review_log.csv')
if not any(r['review_id']=='20261102_007_winner_side' for r in log):log.append(dict(review_id='20261102_007_winner_side',date='2026-11-02',candidate_id=r['candidate_id'],field='winner_camera_side',value='far',reviewer='human_user',evidence='007_score_point_context_h264.mp4; user response 奥',prior_value='unreviewed',scope='winner side only; score-team mapping derived separately'))
write('human_review_log.csv',log)
# Keep machine proposals entirely separate from the reviewed ledger.
teams={'top':'橋場柊一郎・菊山太陽','bottom':'上岡俊介・丸山海斗'}
starts=[1,5,15,21,26,33,39,44,53];boundaries=set(starts[1:]);samples={r['candidate_id'] for r in read('score_suggestion_validation_sample.csv')}
out=[];triage=[]
for r in labels:
 cid=r['candidate_id'];n=int(cid[-3:]);s=by[cid];row=s['suggested_scoreboard_row'];g=max(i for i,v in enumerate(starts) if n>=v);changes=(g+1)//2;phase='normal_game';bottom='near' if changes%2==0 else 'far'
 if g==8:
  phase='final_game';digits=s['score_start'].split('-')
  if len(digits)==2 and all(x.isdigit() for x in digits):
   changes=4+(sum(map(int,digits))+2)//4;bottom='near' if changes%2==0 else 'far'
  else:bottom=''
 side=bottom if row=='bottom' else ('far' if bottom=='near' else 'near') if row=='top' and bottom else ''
 reviewed=r['winner_camera_side_review_status']=='human_confirmed'
 proposal=dict(candidate_id=cid,score_row_proposal=row,winner_team_proposal=teams.get(row,''),winner_side_rule_proposal=side,proposal_status='machine_provisional_not_ground_truth' if row else 'unresolved_no_simple_increment',score_before=s['score_before_end'],score_after=s['score_after_1s'],mapping_basis='game_start_checkpoints_plus_end_change_rule',point_linkage_status='not_independently_validated',human_winner_side=r['winner_camera_side'] if reviewed else '',human_side_review_status=r['winner_camera_side_review_status'],legacy_window_status=r['legacy_window_status'],game_phase=phase)
 out.append(proposal)
 reasons=[]
 if not row:reasons.append('ambiguous_score_change')
 if n in boundaries or n+1 in boundaries:reasons.append('game_boundary_context')
 if n in [55,59,63]:reasons.append('final_game_end_change_context')
 if cid in samples:reasons.append('preselected_validation_sample')
 if reasons:triage.append(dict(candidate_id=cid,priority='1' if 'ambiguous_score_change' in reasons else '2' if any('context' in x for x in reasons) else '3',reasons=';'.join(reasons),human_side_reviewed=str(reviewed).lower(),score_row_and_point_linkage_review='pending',existing_clip=f'{n:03}_score_validation_h264.mp4' if cid in samples else f'uncertain_{n:03}_h264.mp4' if not row else '',scope='review triage only; not an accuracy evaluation sample'))
write('batch_outcome_proposals.csv',out);write('outcome_review_priority.csv',sorted(triage,key=lambda r:(int(r['priority']),r['candidate_id'])))
print('Human side labels:',sum(bool(r['human_winner_side']) for r in out));print('Machine proposal statuses:',dict(Counter(r['proposal_status'] for r in out)));print('Triage:',dict(Counter(r['priority'] for r in triage)));print('Simple proposals without human side:',sum(bool(r['score_row_proposal']) and not r['human_winner_side'] for r in out))
