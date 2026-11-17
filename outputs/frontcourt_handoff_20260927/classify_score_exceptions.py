import csv
from pathlib import Path
p=Path(__file__).resolve().parent
spec=[('011','3-3','blank-A','bottom','advantage_display','Other point field suppressed; bottom advantage identifies score transition, not independent point linkage'),('012','blank-A','4-4','top','advantage_to_deuce','Top inferred from bottom advantage returning to deuce'),('013','4-4','5-4','top','digit_above_template_range','5 absent from old templates'),('014','5-4','6-4','top','digit_above_template_range','6 absent from old templates; game-winning context requires linkage check'),('032','3-4','3-5','bottom','digit_above_template_range','5 absent from old templates'),('052','4-3','5-3','top','digit_above_template_range','5 absent from old templates'),('061','4-4','4-5','bottom','digit_above_template_range','5 absent from old templates'),('062','4-5','5-5','top','digit_above_template_range','5 absent from old templates'),('063','5-5','6-5','top','digit_above_template_range','6 absent from old templates; match not over at this score'),('025','games2-1_points0-4','games2-2_points0-1','','multiple_point_context','Game reset plus following point; do not force a single outcome'),('048','2-2','3-2','top','score_updated_before_candidate_end','Transition precedes nominal end; candidate timing must be checked'),('050','3-3','3-3','','no_score_change_observed','No result linkage established; not proof no point occurred')]
rows=[]
for n,a,b,row,cat,note in spec:rows.append(dict(candidate_id='match1_rally_window_'+n,display_start_visually_read=a,display_end_plus_1s_visually_read=b,score_row_transition_proposal=row,exception_category=cat,evidence=f'exception_scores_{0 if int(n)<=14 else 1 if int(n)<=50 else 2}.jpg',reviewer='Codex image review 2026-11-02',status='ai_score_display_review_only_not_human_outcome',point_linkage_status='pending_motion_review',notes=note))
with (p/'exception_score_display_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(p/'EXCEPTION_SCORE_FINDINGS.md').write_text('''# 得点表示の例外12件の切り分け

2026-11-02、既存の全候補で同じ規則により抽出した開始時・終了1秒後のスコア画像を目視した。人手によるラリー勝者の正解ではなく、AIによる放送表示の読取である。

9件（011/012/013/014/032/052/061/062/063）は従来認識器の対象外表示を含む。5・6・Aを0〜4の数字認識器へ入力していた。011は3-3から下段A（上段ポイント欄は空白）、012は下段Aから4-4への復帰。013以降の該当7件は5または6の表示を含む。候補表示の増点側はCSVに記録したが、映像上の得点事象との1対1対応をまだ確定しない。

025はゲームのリセットと次のポイントへの進行を含みうる。ここは認識器の数字種類を増やすだけでは解決しない。048は2-2→3-2という表示を読めるが、表示更新が候補終了より前。050は3-3が維持され、結果との対応が未確認。これら3件を区間対応の重点監査対象とする。

次は025/048/050の主カメラ区間と放送更新の時刻を照合する。ほかの9件も正解ラベルとして自動採用せず、検証標本との照合を残す。既存16件の確率標本には今回AIが読んだ表示を人手回答として転記しない。005で既に得点提案を開示したため、以後の同じ確認者による独立したチーム名・表示行評価では開示履歴の影響も区別する。これまでの側回答のみで認識器精度を算出しない。

batch_outcome_proposals.csvの元提案51件は、従来の0〜4認識器の出力として保持した。例外表で読み取った10件の表示変化を追加して「61ラリーの勝者確定」とは呼ばない。真のポイント境界・重複・編集・リプレイを調べる前の63候補を63独立ラリーと断定しない。
''',encoding='utf-8')
print('12 exceptions classified: 9 unsupported score displays; 3 temporal/point-linkage issues. Reviewed outcome ledger unchanged.')
