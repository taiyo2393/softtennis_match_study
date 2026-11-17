import csv
from pathlib import Path
p=Path(__file__).resolve().parent
rows=[dict(candidate_id='match1_rally_window_025',observation='994s dissolve shows old game2-1 points0-4; 998s game2-2 points0-0; 1000s service preparation; 1002-1010s play; 1012s points0-1',hypothesis='stale_start_score_before_service_not_proven_multiple_point_mixture',score_row_candidate='bottom',boundary_action='retain original candidate; propose pre-serve scoreboard baseline after reset',status='ai_sparse_context_review_human_motion_pending'),dict(candidate_id='match1_rally_window_048',observation='Play visible through2122s; 2124s points3-2; closeup2126s; original candidate end2125.5s',hypothesis='broadcast_score_update_before_camera_exit_is_compatible_with_single_point',score_row_candidate='top',boundary_action='do not use camera-exit time as point-ending time',status='ai_sparse_context_review_human_motion_pending'),dict(candidate_id='match1_rally_window_050',observation='points3-3 and play visible2258-2286s; 2290s points4-3; candidate050ends2275s and051begins2276.5s',hypothesis='050_and_051_possible_single_point_split',score_row_candidate='top',boundary_action='review 050/051 continuity before assigning two rally IDs',status='ai_sparse_context_review_human_motion_pending')]
with (p/'three_boundary_context_findings.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(p/'THREE_BOUNDARY_CONTEXT_FINDINGS.md').write_text('''# 025・048・050の前後文脈の再確認

旧動画から時刻を明示した16:9全画面の疎な画像列を作成して比較。静止画列によるAI所見であり、連続映像での独立人手判定ではない。元の候補境界・確認済みラベルは変更しない。

025：994秒はカメラ切替のディゾルブ中で旧ゲームの表示。998秒にはゲーム2-2、ポイント0-0となり、その後サーブとプレー、1012秒には下段0→1。従来の「終了後に次のポイントを拾った危険」という説明だけでは不十分で、開始時の表示遅れが有力な代替説明。勝者候補は下段だがまだ確定しない。

048：2122秒にはプレー、2124秒には上段2→3、2126秒は近接映像。得点表示がカメラ退出境界2125.5秒より先に更新されること自体は矛盾ではない。候補窓と1ポイントが対応する可能性を保持する。得点成立の正確な時刻と表示更新は別物。

050/051：050は2258〜2275秒、051は2276.5〜2288.5秒。拡張画像では2258〜2286秒に同じ主カメラと3-3表示のプレーが見え、2290秒に上段4-3。050単体の終了後8秒まで表示が変わらないことは、結果不明の根拠として不十分。1ラリーが2候補に分割された可能性がある。まず境界付近を連続映像で確認し、同一点なら候補を統合したpoint_idを別表で提案する。元CSVを削除・書換してなかったことにはしない。

重要な含意：63候補をそのまま63独立ポイントと扱わない。同じ結果を二重に数えない。従来のスコア候補精度とラリー単位結果精度を分ける。
''',encoding='utf-8')
print('Three revised context hypotheses saved; labels and original boundaries unchanged')
