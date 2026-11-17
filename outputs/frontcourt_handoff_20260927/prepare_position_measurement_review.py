import csv,json
from pathlib import Path
p=Path(__file__).resolve().parent
rows=[]
for n,f in [('044',115403),('056',154116)]:
 for slot in ['near_front','near_rear','far_front','far_rear']:
  rows.append(dict(candidate_id=n,source_frame=f,player_slot=slot,left_foot_x_px='',left_foot_y_px='',right_foot_x_px='',right_foot_y_px='',foot_visibility='',support_status='',ground_plane_position_status='',reviewer='',review_status='unreviewed',evidence=f'{n}_position_annotation_raw.jpg'))
path=p/'independent_footpoint_annotation_template.csv'
if not path.exists():
 with path.open('w',encoding='utf-8-sig',newline='') as s:w=csv.DictWriter(s,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
findings=[dict(candidate_id='044',frame=115403,slot='near_rear',finding='One foot raised; supporting foot appears visible',status='ai_visual_provisional'),dict(candidate_id='056',frame=154116,slot='near_front',finding='Both feet may be raised; ground contact cannot be assumed from box bottom',status='ai_visual_provisional'),dict(candidate_id='056',frame=154116,slot='far_rear',finding='Near-front player occludes far player; one foot visible, other foot needs temporal review',status='ai_visual_provisional')]
with (p/'footpoint_measurement_risks.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=findings[0]);w.writeheader();w.writerows(findings)
(p/'NEXT_REVIEW.md').write_text('''# 次回の確認事項

## ここまでの到達点
044・056の得点側は人手で手前と確認。チーム名は得点表示と試合開始時の側対応・ゲームのチェックポイント照合から、それぞれ上岡・丸山、橋場・菊山と導出。終了事象は044アウト、056の2バウンド。有効サービス、候補時点の4人選択、2つの再連結先の同一人物性も人手確認済み。

再連結2件の成功候補を確認しただけで、誤連結率や全区間のID精度は算出できない。補間位置を観測・正解と扱わない。

## 時刻の不確実性の予備評価
044はF115395〜115403の9ソースフレーム、056はF154111〜154116の6フレームを、既存接触範囲に30フレームを加えた採取候補として固定。後から有利な時点を選ばない。全時点で同じ4トラッカーIDを取得できたが、その間の全画像での本人同定は独立未確認。

検出枠下端からの正規化座標で、ペア間隔の変動幅は044の左右でコート幅の最大0.90%、前後でコート長の最大1.70%。056は左右の最大0.33%、前後の最大0.63%。小さい変動が正しい座標を意味するわけではなく、接地点誤差や校正誤差は含めて評価できていない。

左右差はコート幅、前後差はコート長で別々に正規化。両者の正規化値をそのまま足したユークリッド距離は使用していない。物理距離を出すなら軸の実寸と校正を確認する。

## 位置測定で先に解決する問題
候補4人が正しく選ばれても、箱下端が地面上の選手位置とは限らない。056の手前前衛スロットは両足が浮いている可能性があり、奥の橙色選手は足元の一部を遮蔽される。これらはAIの画像所見であり、人手未確認。

足の代表点を両足の中点、支持足の接地点、身体の地面投影のどれと定義するかで値は変わる。定義をラリーごとに都合よく切り替えない。まず左右の足の画像位置と支持状態を独立に注釈し、両足が浮く／遮蔽される場合の観測可否を明記する。足が浮いている位置を地面ホモグラフィへ入れると偏りうるので、無条件に地上座標の正解としない。

独立注釈用の原画像044_position_annotation_raw.jpg / 056_position_annotation_raw.jpgには検出点を描かない。independent_footpoint_annotation_template.csvの座標・支持状態は空欄を保持。候補との差を見る前に独立に記入する。footpoint_candidate_review画像は候補点を黄色で示す診断用で、独立正解の作成には使わない。

## 再取得映像
再取得版にも056の同一サーブ場面でほぼ重複した隣接フレームがある。旧動画は以前とハッシュ一致し、元注釈を維持。新動画への時刻変換は局所候補にとどめ、全体へ一括移行しない。

## 次回の最小の人手確認
056で両足が浮いて見える手前選手について、前後の短い映像で支持状態を確認する。その後、足の位置定義を固定して4選手×2ラリーの独立画像注釈を行う。終了の正確な1フレームはサーブ後1時点の配置研究の先行必須条件にしない。終了事象の種類を厳密時刻に読み替えない。

研究上は予備測定2ラリーの段階。勝敗との相関、予測性能、方法の新規性、戦術推奨は未評価。
''',encoding='utf-8')
print('Independent annotation template and concrete next-review handoff saved')
