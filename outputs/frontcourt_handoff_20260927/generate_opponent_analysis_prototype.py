import csv,json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent

def read(n):
 with (p/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
points=read('exploratory_position_outcome_points.csv');features={r['candidate_id']:r for r in read('exploratory_position_outcome_features.csv')};labels={r['candidate_id'][-3:]:r for r in read('rally_label_review.csv')};A='上岡俊介・丸山海斗';B='橋場柊一郎・菊山太陽';records=[]
for n,f in features.items():
 winteam=f['winner_team'];other=B if winteam==A else A
 for side in ['near','far']:
  pair=[r for r in points if r['candidate_id']==n and r['camera_side']==side];team=winteam if side==f['winner_side'] else other;row=dict(candidate_id=n,team=team,camera_side=side,role='serve' if side==f['server_side_candidate'] else 'receive',source_frame=f['source_frame'],pair_positions_available=len(pair),measurement_status='AI_approximate_contact_time_uncertainty_retained',lateral_gap=None,longitudinal_gap=None,team_view_center_x=None,team_view_left_margin=None,team_view_right_margin=None)
  if len(pair)==2:
   xs=[float(r['x_normalized']) if side=='near' else 1-float(r['x_normalized']) for r in pair];ys=[float(r['y_normalized']) if side=='near' else 1-float(r['y_normalized']) for r in pair]
   row.update(lateral_gap=abs(xs[0]-xs[1]),longitudinal_gap=abs(ys[0]-ys[1]),team_view_center_x=statistics.mean(xs),team_view_left_margin=min(xs),team_view_right_margin=1-max(xs))
  row['clip']=next((name for name in [f'{n}_score_point_context_h264.mp4',f'{n}_full_point_context_h264.mp4',f'{n}_full_score_context_h264.mp4',f'{n}_finish_score_context_h264.mp4',f'{n}_first_point_side_anchor_h264.mp4'] if (p/name).exists()),'')
  records.append(row)
structured=[];lines=['# 対戦相手の配置分析：文章出力の試作','', '既存の1試合・11候補を使った動作確認です。新規動画の自動取込や、作戦の有効性評価が完成したものではありません。サーブ／レシーブはAIの暫定区分、位置は箱下端の概算値です。得点との関係から作戦効果を断定しません。','']
for team in [A,B]:
 lines.extend([f'## {team}',''])
 for role,title in [('serve','サーブ時'),('receive','レシーブ時')]:
  subset=[r for r in records if r['team']==team and r['role']==role];valid=[r for r in subset if r['lateral_gap'] is not None];assert valid
  metric={k:statistics.median(r[k] for r in valid) for k in ['lateral_gap','longitudinal_gap','team_view_center_x','team_view_left_margin','team_view_right_margin']}
  text=f'{title}の対象{len(subset)}場面中、ペア2人の概算位置が揃うのは{len(valid)}場面です。左右間隔の中央値はコート幅の{metric["lateral_gap"]*100:.1f}%、前後間隔はコート長の{metric["longitudinal_gap"]*100:.1f}%でした。自陣からネットを向いた左右に統一したペア中心の中央値は、左端0・右端100として{metric["team_view_center_x"]*100:.1f}です。'
  lines.extend([f'### {title}',text,'','**作戦候補の扱い：** この集計だけでは特定コースの通過可能性や配球成功率が分からないため、狙うコースの断定は保留します。配置が実際にどちらへ寄るかを根拠映像で確認し、配球候補を検討する材料として使います。','', '**根拠映像：**'])
  for r in valid:
   cid=r['candidate_id'];clip=r['clip'];lines.append(f'- 候補{cid}・旧動画F{r["source_frame"]}：[{clip or "映像未登録"}]({(p/clip).resolve().as_posix()})' if clip else f'- 候補{cid}・旧動画F{r["source_frame"]}：映像未登録')
  lines.append('');structured.append(dict(team=team,role=role,total_contexts=len(subset),usable_pair_contexts=len(valid),medians=metric,observation_text=text,evidence=[dict(candidate_id=r['candidate_id'],source_frame=r['source_frame'],clip=r['clip']) for r in valid],recommendation_status='abstain_effectiveness_unverified',claim_kind='descriptive_observation_only'))
lines.extend(['## 制約と次の機能','', '左右の偏りはサーブの左右で変わりうるため、この少数例を相手固有の弱点とは呼びません。次はボールの打球区間から実際の配球方向を付け、配置と配球が対応する場面を根拠として追加します。勝敗に都合のよい場面だけで文章を作らず、欠測を残します。'])
(p/'opponent_analysis_prototype.md').write_text('\n'.join(lines),encoding='utf-8');(p/'opponent_analysis_prototype.json').write_text(json.dumps(structured,ensure_ascii=False,indent=2),encoding='utf-8')
assert len(records)==22 and len(structured)==4
assert sum(r['pair_positions_available']==2 for r in records)==20
assert all(r['evidence'] and r['usable_pair_contexts']==len(r['evidence']) for r in structured)
print(json.dumps([dict(team=r['team'],role=r['role'],n=r['usable_pair_contexts'],total=r['total_contexts'],medians=r['medians']) for r in structured],ensure_ascii=False,indent=2))
