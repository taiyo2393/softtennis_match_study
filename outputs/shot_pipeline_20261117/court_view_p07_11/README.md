# match2 第7〜11点：奥側の前後位置補間・固定済み人物対応と打者判定

tracking_v2を読み取って再描画した。Bの入力には従来Aの平滑化前位置を維持し、Bで人物を固定した後、最終Aの奥側位置だけを下記の規則で再計算する。Bの対応付け・Cの球と手首による打者判定は変更しない。打球の時刻・側・ラリーは本人ラベル32行。人物名、位置、球候補、打者は自動処理で未確認。N1/N2/F1/F2は各ラリーの初期スロットに従う人物名で、ラリー中は固定し、役割名としては扱わない。

## 再実行

```powershell
& study\.venv\Scripts\python.exe -B -X utf8 outputs\shot_pipeline_20261117\court_view_p07_11\build_court_view.py --far-depth-only
& study\.venv\Scripts\python.exe -B -X utf8 outputs\shot_pipeline_20261117\court_view_p07_11\far_depth_revision.py --verify
```

今回は保存済み球候補をそのまま使用する。生成時の条件は、study/third_party/racketvision-env/Scripts/python.exeからrun_ball_candidates.pyを実行し、run_match2_first_point_baseline.run_baselineを呼び出したもの。チェックポイントのSHA-256も第1点のprotocolと照合済み。BGR・JPEG品質95、512×288、背景中央値120枚、t−3〜t、CUDA AMP、batch=4、閾値0.3、最大成分を使用。背景120枚は各表示区間内から等間隔に選ぶ。最初の3Fの文脈も元動画から取得。元フレーム番号はffmpegの元PTSと全件照合する。

## A：下が隠れた枠

far_*だけを変更し、保存済みB/C・手前側をそのまま使用する。同じFの手前枠との横重なりが奥枠幅の10%以上、かつ手前y1<奥y2なら下隠れ。高さの比は遮蔽判定に使わない。

下隠れでは、そのFの枠中心xと、同一人物・同一ラリーの前後の下隠れでない採用点の画像y2を時間で線形補間したyを使う。区間長は無制限。片側しかなければその側を保持、両側ともなければ欠測。trackingと同じ射影を適用し、速度採用時はobserved_x_depth_interpolated。アンカーは速度採用結果が一致するまで更新し、最終採用点のみを使う。

下隠れでないFは枠下端を直接使用し、h>1.35×0.73×(y2−415)の大きな枠だけを高さ棄却する。地平線415px、k=0.73は全点で指定値のまま使用し、再推定なし。奥側の高さリセット・80%点基準・頭基準を廃止。従来の速度規則・±8元F中央値平滑化・採用点間30元F以内の通常補間は維持する。

左の下隠れ枠は実線で名前に「下隠れ」。右の塗り円＋白縁は「横は観測・前後は補間」、白抜き円は全座標の通常補間。表示は中央値平滑化後である。

点・奥側選手ごとの全status件数はfar_status_counts.csv（表示区間内の全元F／全出力範囲）、通常補間前の棄却内訳はfar_filter_status_counts.csv。前回比較表・範囲外件数は[修正報告](far_depth_revision_report.md)。


## B：人物のつなぎ直し

従来Aで補正した平滑化前の代理位置から同側2人に対応付ける。この入力と対応付け結果は今回の最終A改訂から切り離して維持する。同じtrack_idはラリー内で同じ人、同時刻の別trackは別人。新trackは既存の人に対して間隔1秒以内、距離8m/s×間隔＋1.5m以下で接続。同時2人なら接続可能数を最大化した上で合計距離を最小にする。接続不可は欠測として記録し、3人目を作らない。ラリー境界で名前を初期化する。姿勢の手首系列も、その枠の人物へ付け替える。初回の2人目が欠測なら初出スロットで初期化する。

下表は元スロットへ新しいtrackが入り、人物固定によって別名へ戻した区間の開始。前trackはその元スロット名に固定した人物の直前track。全track接続と未接続観測はidentity_summary.json、毎Fの元スロット対応はidentity_map_p*.csv。

|点|ラリー|F|元の名前|前track→入ってきたtrack|固定後の名前|
|---|---:|---:|---|---|---|
|11|1|32930|F1|30→4|F2|
|11|1|32932|F2|4→30|F1|

## C：球候補と打者

球は各ラリー最初の打球−60F〜最後＋90Fを毎F推定。左の橙の小印は「球：自動検出（未確認）」であり、056の確認済みの緑丸とは区別する。球は画像座標のまま扱い、地面へ射影しない。

打球±3F内の候補から、各人の横に幅30%・上に高さ30%広げた枠への距離を求め、各人の最小値を比較。片方が0かつ他方が他方の元枠幅以上、または近/遠の比が0.5未満なら球の打者。各最小距離を得たF、距離、比、全候補の座標・信頼度・両距離をCSVに保存。

手首は打球−8〜+4Fの最大速度、両人に有効値があり一意の最大が他方の1.3倍以上。球と手首が一致ならauto_ball_wrist、球のみ決まればauto_ball、球候補がない場合のみ手首だけをauto_wrist。食い違い・球候補が曖昧・双方なしはundecidedとし、双方の候補と理由を残す。

球候補がある打球は24/32。判定内訳：{'auto_ball_wrist': 18, 'auto_ball': 2, 'undecided': 7, 'auto_wrist': 5}。これは検出の利用可能性と自動判定件数であり、精度ではない。

|点|F|球候補F数|距離1/2 px|球|手首|最終|status|理由|
|---|---:|---:|---|---|---|---|---|---|
|7|22848|5|0.00/176.02|F1|F1|F1|auto_ball_wrist|ball_wrist_agree|
|7|22911|7|445.52/0.00|N2|N2|N2|auto_ball_wrist|ball_wrist_agree|
|7|23482|4|0.00/243.40|F1|F1|F1|auto_ball_wrist|ball_wrist_agree|
|7|23565|2|264.59/0.00|N2|N2|N2|auto_ball_wrist|ball_wrist_agree|
|7|23650|1|0.00/338.12|F1|F?|F1|auto_ball|ball_only|
|7|23803|0|欠測/欠測|N?|N?|N?|undecided|no_decisive_evidence|
|7|23906|7|46.49/0.00|F2|F2|F2|auto_ball_wrist|ball_wrist_agree|
|7|23934|1|0.00/欠測|N?|N?|N?|undecided|no_decisive_evidence|
|8|25246|1|10.16/317.64|F1|F1|F1|auto_ball_wrist|ball_wrist_agree|
|8|25321|7|451.73/3.68|N2|N2|N2|auto_ball_wrist|ball_wrist_agree|
|8|25400|1|0.23/欠測|F?|F1|F?|undecided|ball_candidates_ambiguous|
|8|25443|4|0.00/0.00|N?|N2|N?|undecided|ball_candidates_ambiguous|
|8|25535|2|0.70/98.13|F1|F?|F1|auto_ball|ball_only|
|8|25652|7|22.40/267.09|N1|N1|N1|auto_ball_wrist|ball_wrist_agree|
|8|25787|2|35.20/161.45|F1|F1|F1|auto_ball_wrist|ball_wrist_agree|
|8|25863|0|欠測/欠測|N?|N?|N?|undecided|no_decisive_evidence|
|9|27464|2|1.91/153.57|F1|F1|F1|auto_ball_wrist|ball_wrist_agree|
|9|27976|7|0.00/212.09|F1|F1|F1|auto_ball_wrist|ball_wrist_agree|
|9|28069|1|0.00/610.43|N1|N1|N1|auto_ball_wrist|ball_wrist_agree|
|9|28152|3|0.00/320.80|F1|F1|F1|auto_ball_wrist|ball_wrist_agree|
|9|28292|1|0.00/708.17|N1|N1|N1|auto_ball_wrist|ball_wrist_agree|
|9|28375|2|0.00/287.13|F1|F1|F1|auto_ball_wrist|ball_wrist_agree|
|9|28526|0|欠測/欠測|N?|N?|N?|undecided|no_decisive_evidence|
|9|28597|0|欠測/欠測|F?|F2|F2|auto_wrist|no_ball_candidates|
|10|30452|0|欠測/欠測|F?|F1|F1|auto_wrist|no_ball_candidates|
|10|30548|5|0.00/255.28|N1|N1|N1|auto_ball_wrist|ball_wrist_agree|
|10|30583|5|0.56/0.00|F2|F1|F?|undecided|ball_wrist_conflict|
|10|30613|1|0.00/378.45|N1|N1|N1|auto_ball_wrist|ball_wrist_agree|
|11|32755|0|欠測/欠測|F?|F1|F1|auto_wrist|no_ball_candidates|
|11|32844|0|欠測/欠測|N?|N1|N1|auto_wrist|no_ball_candidates|
|11|33001|0|欠測/欠測|F?|F2|F2|auto_wrist|no_ball_candidates|
|11|33041|5|108.56/0.00|N2|N2|N2|auto_ball_wrist|ball_wrist_agree|

### 前回の判定からの変更

比較元はhitter_auto_before_abc.csv。前回と名前またはstatusが変わったものを全件記載。

|点|F|変更前|変更後|
|---|---:|---|---|
|7|22848|F1 (auto_wrist)|F1 (auto_ball_wrist)|
|7|22911|N2 (auto_wrist)|N2 (auto_ball_wrist)|
|7|23482|F1 (auto_wrist)|F1 (auto_ball_wrist)|
|7|23565|N2 (auto_wrist)|N2 (auto_ball_wrist)|
|7|23650|F? (undecided)|F1 (auto_ball)|
|7|23906|F2 (auto_wrist)|F2 (auto_ball_wrist)|
|8|25246|F1 (auto_wrist)|F1 (auto_ball_wrist)|
|8|25321|N2 (auto_wrist)|N2 (auto_ball_wrist)|
|8|25400|F1 (auto_wrist)|F? (undecided)|
|8|25443|N2 (auto_wrist)|N? (undecided)|
|8|25535|F? (undecided)|F1 (auto_ball)|
|8|25652|N1 (auto_wrist)|N1 (auto_ball_wrist)|
|8|25787|F1 (auto_wrist)|F1 (auto_ball_wrist)|
|9|27464|F1 (auto_wrist)|F1 (auto_ball_wrist)|
|9|27976|F1 (auto_wrist)|F1 (auto_ball_wrist)|
|9|28069|N1 (auto_wrist)|N1 (auto_ball_wrist)|
|9|28152|F1 (auto_wrist)|F1 (auto_ball_wrist)|
|9|28292|N1 (auto_wrist)|N1 (auto_ball_wrist)|
|9|28375|F1 (auto_wrist)|F1 (auto_ball_wrist)|
|10|30548|N1 (auto_wrist)|N1 (auto_ball_wrist)|
|10|30583|F1 (auto_wrist)|F? (undecided)|
|10|30613|N1 (auto_wrist)|N1 (auto_ball_wrist)|
|11|33001|F1 (auto_wrist)|F2 (auto_wrist)|
|11|33041|N2 (auto_wrist)|N2 (auto_ball_wrist)|

### 前回未決定の6打球

|点|F|球|手首|結果|status|理由|
|---|---:|---|---|---|---|---|
|7|23650|F1|F?|F1|auto_ball|ball_only|
|7|23803|N?|N?|N?|undecided|no_decisive_evidence|
|7|23934|N?|N?|N?|undecided|no_decisive_evidence|
|8|25535|F1|F?|F1|auto_ball|ball_only|
|8|25863|N?|N?|N?|undecided|no_decisive_evidence|
|9|28526|N?|N?|N?|undecided|no_decisive_evidence|

## 動画・確認画像・検証

1920×1080、15fps、無音。通常は元映像2Fごと。指定した奇数Fも厳密に収録するため必要な場合だけ1コマ追加する。各ラリー範囲を連結し、対応をvideo_source_frame_map_p*.csvに保存。

|点|動画|コマ数|秒|全復号|
|---|---|---:|---:|---|
|7|[MP4](point_07.mp4)|409|27.267|True|
|8|[MP4](point_08.mp4)|384|25.600|True|
|9|[MP4](point_09.mp4)|642|42.800|True|
|10|[MP4](point_10.mp4)|157|10.467|True|
|11|[MP4](point_11.mp4)|219|14.600|True|

![check_abc_p07_F23422.jpg](check_abc_p07_F23422.jpg)

![check_abc_p10_F30633.jpg](check_abc_p10_F30633.jpg)

![check_abc_p11_F32999.jpg](check_abc_p11_F32999.jpg)

## 限界

下が隠れている間の前後位置は観測ではなく補間であり、その間の前後への動きは見えない。片側のみの場合は一定値保持である。前後の枠下端自体の誤差も引き継ぐ。固定式による大きな枠の棄却は、2人のまとまりを完全に識別するものではない。位置精度は独立した正解で未検証。人物対応は前回のまま固定し、同一ID内部の人物混同や誤接続は修復しない。

球候補は自動出力で誤検出を含み、枠に近いだけで接触は証明できない。球と手首の一致も独立した正解ではない。前回未決定の打球が決まっても正解率向上とは結論できない。位置・人物対応・打者の評価には映像による独立した参照ラベルが必要。模式線は打者間の説明用で球軌道・バウンド・3D位置ではない。

tracking/、tracking_v2/、pose/は読み取り専用で入力SHA-256を前後照合。ball_quality_*への書き込みは行わない。git commitは行わない。256KiB超ファイルと画像・動画のgit除外をgit check-ignoreで検証し、file_inventory.csvに保存する。球キャッシュは一覧をフォルダ合計にまとめるが、全ファイルのgit除外を検査する。

## 今回の確認と集計

表示区間内の全元フレーム（動画の2F間引きで省いたFも含む）を対象に、最終statusが採用または補間のcourt_yの5%・50%・95%点をmで集計。空間的な切り取りは行わない。移動量は同じラリーの連続する元Fの両方が有効な場合のコート平面ユークリッド距離。欠測やラリー境界はまたがない。単位はm/元フレームで、最大となる遷移の両端Fを記録する。

集計はdisplayed_position_metrics.csvとfar_depth_revision_report.json、全status件数はfar_status_counts.csvを参照。修正直前との比較・B/Cの16ファイルのSHA-256・手前側の全従来列の一致・変更禁止入力の照合も同reportに保存する。

第8点F25470のfar_rearのtrack279は、Bの接続結果を維持するため欠測扱いのまま。右図は前後の採用点による補間（白抜き）で表示する。位置誤差の改善を示すには独立した正解位置が必要。

|点|選手|y5% m|y50% m|y95% m|最大移動 m/元F|元Fの遷移|
|---|---|---:|---:|---:|---:|---|
|7|N1 (near_front)|4.672|5.901|8.160|0.241719|F23682→F23683|
|7|N2 (near_rear)|-3.267|0.102|2.012|0.173863|F23563→F23564|
|7|F1 (far_rear)|22.141|26.466|29.660|0.578903|F23641→F23642|
|7|F2 (far_front)|12.234|13.593|20.943|0.472205|F23894→F23895|
|8|N1 (near_front)|-2.166|-1.360|-0.202|0.144986|F25406→F25407|
|8|N2 (near_rear)|-0.094|7.248|10.607|0.623310|F25432→F25433|
|8|F1 (far_rear)|23.460|24.610|25.911|0.884223|F25257→F25258|
|8|F2 (far_front)|12.295|13.402|16.260|0.396679|F25367→F25368|
|9|N1 (near_front)|-1.879|1.501|4.805|0.160305|F28084→F28085|
|9|N2 (near_rear)|0.147|7.332|9.168|0.286277|F28165→F28166|
|9|F1 (far_rear)|14.361|23.079|24.203|0.423239|F28508→F28509|
|9|F2 (far_front)|15.716|23.809|25.406|0.296097|F27845→F27846|
|10|N1 (near_front)|3.094|4.561|4.771|0.102777|F30570→F30571|
|10|N2 (near_rear)|-0.292|0.886|1.415|0.075190|F30594→F30595|
|10|F1 (far_rear)|12.242|16.799|23.225|0.511912|F30593→F30594|
|10|F2 (far_front)|16.003|16.449|18.075|0.250900|F30615→F30616|
|11|N1 (near_front)|-2.217|3.071|4.941|0.287748|F32898→F32899|
|11|N2 (near_rear)|0.706|2.389|10.960|0.602840|F33037→F33038|
|11|F1 (far_rear)|14.274|20.257|23.298|0.505868|F32841→F32842|
|11|F2 (far_front)|14.926|15.639|30.578|0.899526|F33024→F33025|
