import csv
from pathlib import Path
p=Path(__file__).resolve().parent

def read(name):
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);return list(rd.fieldnames),list(rd)
def write(name,fields,rows):
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
fields,labels=read('rally_label_review.csv')
for k in ['point_group_id','point_group_review_status']:
 if k not in fields:
  fields.append(k)
  for r in labels:r[k]=''
for r in labels:
 if r['candidate_id'][-3:] in ['050','051']:
  r['point_group_id']='match1_point_group_050_051';r['point_group_review_status']='human_confirmed_same_rally_partial_context'
  note='2026-11-02 user confirmed 050_051_continuity_review_h264.mp4 contains same rally across candidate boundary. Group 050/051 as one point; exact serve/end boundaries and winner side not confirmed by this answer.'
  if note not in r['review_note']:r['review_note']=(r['review_note']+' | '+note).strip(' |')
write('rally_label_review.csv',fields,labels)
fields,log=read('human_review_log.csv')
if not any(r['review_id']=='20261102_050_051_same_rally' for r in log):log.append(dict(review_id='20261102_050_051_same_rally',date='2026-11-02',candidate_id='match1_rally_window_050;match1_rally_window_051',field='candidate_point_linkage',value='same_rally',reviewer='human_user',evidence='050_051_continuity_review_h264.mp4; user response 同じラリー',prior_value='ai_split_point_hypothesis',scope='continuity across candidate boundary only; winner, exact service/end frames and tracking not confirmed'))
write('human_review_log.csv',fields,log)
mapping=[]
for r in labels:
 merged=r['candidate_id'][-3:] in ['050','051'];mapping.append(dict(candidate_id=r['candidate_id'],point_group_id='match1_point_group_050_051' if merged else r['candidate_id'].replace('rally_window','point_group_candidate'),group_status='human_confirmed_same_rally' if merged else 'singleton_unverified_point_linkage',exact_point_boundaries_status='pending',evidence='050_051_continuity_review_h264.mp4; human response 同じラリー' if merged else 'legacy candidate only'))
write('candidate_point_group_map.csv',list(mapping[0]),mapping)
# Collapse proposals by point group; retain conflicts and never upgrade AI to human.
_,proposals=read('batch_outcome_proposals.csv');by={r['candidate_id']:r for r in proposals};groups={}
for m in mapping:groups.setdefault(m['point_group_id'],[]).append(m['candidate_id'])
out=[]
for gid,cids in groups.items():
 teams={by[c]['winner_team_proposal'] for c in cids if by[c]['winner_team_proposal']};human={by[c]['human_winner_side'] for c in cids if by[c]['human_winner_side']}
 out.append(dict(point_group_id=gid,candidate_ids=';'.join(cids),candidate_count=len(cids),winner_team_machine_proposal=next(iter(teams)) if len(teams)==1 else '',winner_team_proposal_status='conflict' if len(teams)>1 else 'machine_provisional' if teams else 'unknown',winner_side_human=next(iter(human)) if len(human)==1 else '',point_linkage_status='human_confirmed_same_rally' if len(cids)>1 else 'singleton_unverified',exact_boundaries_status='pending'))
assert len(mapping)==63 and len(out)==62
write('point_group_outcome_proposals.csv',list(out[0]),out)
(p/'POINT_GROUPING_NOTES.md').write_text('''# 候補050・051の同一ラリー確認

2026-11-02、ユーザーが境界前後の連続映像を「同じラリー」と判定。050と051をmatch1_point_group_050_051へ対応づけた。元候補63行と元候補時刻は維持し、開始・終了フレームを推測で確定しない。

以後はcandidate_point_group_map.csvの対応でポイント群へ集約する。今回の統合だけを反映すると63候補は62群となる。ただし他の候補が1対1で真のポイントに対応するかは未検証であり、「62ラリー確定」ではない。

point_group_outcome_proposals.csvは同一群に重複得点を持たせない集約表。050は単純な表示変化なし、051は上段3→4の機械候補であるため、統合群へ上段のチーム提案を1件だけ保持。今回人手が答えたのは連続性で、勝者ではない。勝者・得点側は人手確認済みへ昇格しない。

機械候補を再生成した後はrecord_050_051_same_rally.pyを再実行して集約表を更新する。元の63候補単位で得点率や独立標本数を計算しない。
''',encoding='utf-8')
print('63 candidate rows preserved; 050/051 human-confirmed shared group; 62 provisional groups; no winner labels promoted')
