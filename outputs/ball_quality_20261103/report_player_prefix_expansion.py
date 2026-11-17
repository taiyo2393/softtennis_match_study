"""Render a construction diagnostic and concise Japanese report from verified data."""
import csv,json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'ball_quality_20261109/rally_player_prefix_expansion'
def read(n):
    with (OUT/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    assert json.loads((OUT/'verification.json').read_text(encoding='utf-8'))['status']=='passed'
    meta=read('rally_metadata.csv');obs=read('requested_player_observations.csv');sensitivity=read('contact_timing_sensitivity.csv')
    outcomes={r['rally_candidate_id']:r['winner_camera_side'] for r in read('human_outcomes.csv')}
    values=[]
    for m in meta:
        cid=m['rally_candidate_id'];rows=[r for r in obs if r['rally_candidate_id']==cid];seq=[]
        for d in range(0,61,2):
            subset=[r for r in rows if int(r['relative_source_frame'])==d]
            seq.append(-1 if all(r['missing_reason']=='before_player_export_start' for r in subset) else sum(int(r['observation_present']) for r in subset))
        values.append(seq)
    cmap=ListedColormap(['#cbd5e1','#f8fafc','#fee2e2','#fed7aa','#fde68a','#167d65'])
    fig,ax=plt.subplots(figsize=(12,5.8),layout='constrained')
    ax.imshow(np.array(values),cmap=cmap,norm=BoundaryNorm(np.arange(-1.5,5.5),cmap.N),aspect='auto',interpolation='nearest')
    ax.set_yticks(range(len(meta)),[m['rally_candidate_id'][-3:]+f"   {m['four_observed_times']}/31 four boxes" for m in meta])
    ax.set_xticks([0,5,10,15,20,25,30],[f'{x*2/(60000/1001):.3f}' for x in [0,5,10,15,20,25,30]])
    ax.set_xlabel('Seconds after fixed input-window start (last possible contact +30 source frames)')
    ax.set_title('Available player observations in a fixed early window\nCounts describe export availability, not identity or position accuracy',pad=14)
    ax.set_xticks(np.arange(-.5,31,1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.5);ax.tick_params(which='minor',bottom=False,left=False)
    legend=[Patch(color=cmap(i+1),label=f'{i} observations') for i in range(5)]+[Patch(color=cmap(0),label='Before export starts')]
    ax.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3,frameon=False)
    fig.savefig(OUT/'observation_availability.png',dpi=170);plt.close(fig)
    strata={'ai_service_sequence':'AIによる有効サーブ候補','legacy_manual_contact':'過去の手動接触記録','human_valid_service_ai_timing':'有効性は人手確認・時刻はAI候補'}
    table='\n'.join(f"|{m['rally_candidate_id'][-3:]}|{'手前' if outcomes[m['rally_candidate_id']]=='near' else '奥'}|{strata[m['anchor_stratum']]}|{m['input_first_source_frame']}–{m['input_end_source_frame']}|{m['four_observed_times']}/31|" for m in meta)
    missing=Counter(r['missing_reason'] for r in obs if r['observation_present']=='0')
    lines=[]
    for m in meta:
        cid=m['rally_candidate_id']
        rs=[r for r in sensitivity if r['rally_candidate_id']==cid and r['feature'].endswith('_mean') and r['range_norm']!='']
        peak=max(rs,key=lambda r:float(r['range_norm']))
        lines.append(f"|{cid[-3:]}|{int(m['anchor_last_source_frame'])-int(m['anchor_first_source_frame'])+1}|{peak['feature']}|{float(peak['range_norm']):.6f}|")
    text=f'''# 既存8候補の選手配置入力データ（2026-11-09）

既存の人手得点側と連続選手出力を結び、8候補（手前得点5・奥得点3）を同じ規則で特徴量化した。新しい注釈依頼・モデル学習・得点との関連推定はしていない。データ整形を2例から8候補へ広げた段階であり、8件すべての解析条件が独立に確認されたという結果ではない。

## 今回の入力範囲

各候補のサーブ接触候補範囲の末尾+30フレームを開始とし、2ソースフレーム間隔31時点、約1.001秒を採る。接触末尾から約0.5005〜1.5015秒の窓で、追跡の初期化時刻や見えやすさには合わせない。以前の044・056の「接触直後から約1秒」とは別の窓であり、旧出力は保持した。サーブ時刻の幅によって、実際の接触からの経過時間は候補間で異なる。

|候補|人手得点側|開始の根拠|入力source frame|4人分の観測がある時点|
|---|---|---|---|---|
{table}

4人の箱が出た割合は、本人同定精度・位置精度ではない。選手IDや位置の人手確認の範囲は元の記録を引き継ぎ、全時点の独立正解を得たと扱わない。8候補は同じ試合の既存素材による便宜標本である。

![固定時間窓での選手観測の有無](observation_availability.png)

## 確認済みと暫定の違い

044・056は有効サーブを人手確認済み。接触と終端のフレーム範囲はAI候補のままだが、今回の入力末尾が既存の終端候補より前にある。ほかの6候補では、この出力作成だけで得点成立前と新たに確認したわけではない。カメラ退出時刻やスコア更新を終端に代用していない。

002・003・005・007・048はAIによるサーブ候補を使う探索用入力。048には旧探索範囲より後の修正済みF125859〜125881を採用し、古い粗探索を使わない。025は過去の手動接触時刻1000.601100秒を同じ既存規則で整数化し、F59976を名目値とした。台帳のimpact_confirmedを再利用するが、現在の人手による有効性確認や厳密な誤差幅と同一視しない。

025の先頭4時点（16選手枠）は出力開始前。開始を後ろへ動かさず、before_player_export_startと明示した。全992選手枠のうち元の観測があるものは949、欠測43（出力開始前{missing['before_player_export_start']}、その他{missing['no_exported_observation_not_physical_absence']}）。補間・直前値埋め・ゼロ座標埋めはしていない。出力の欠測は選手の物理的不在を意味しない。

044はこの窓で4人観測15/31時点。以前の窓の29/31より少ないが、異なる時間範囲を使ったことによる値で、追跡の変更・精度低下の推定ではない。良い時刻へ差し替えない。056は今回31/31だが、これも精度向上の主張ではない。

## 入力の表現と検証

元4K画像の検出枠下端中央を画像幅・高さで割り、手前・奥それぞれの2人の重心x/y、左右・上下の絶対差を作った。ペア内のIDを交換しても同じ表現になるが、別人や別コート側への誤割当ては防げない。コート上の距離・足の接地点・3次元速度ではない。

1時点版は指定窓の最後の要求時点を使う。区間版は指定窓内の平均、開始から終了への差、ペア観測率を追加する。開始・終了が欠測なら変化量も欠測にする。得点側・絶対時刻・動画パス・接触根拠・終端候補・全ラリーの長さを入力特徴へ入れていない。元の追跡は未来情報を使う可能性があり、処理全体としてリアルタイム／因果的な予測手順は未実証。

接触候補範囲の各整数フレームを基準に同じ窓を作り直す感度診断も保存した。以下は区間平均8特徴のうち変動幅が最大のもの。画像正規化座標の差で、正解との誤差ではない。025の0は誤差ゼロではなく、元の誤差幅が不明で名目値1点しかないことを示す。欠測構成も変わりうるため、変動を純粋な動作差だけと解釈しない。

|候補|接触候補の整数時点数|最大変動の平均特徴|変動幅|
|---|---|---|---|
{chr(10).join(lines)}

`verification.json`では、元の観測から248時点・992選手枠を照合し、3704特徴値と93個の時間窓を別実装で再計算した。ペア内交換992条件、未来の観測追加8条件も照合。025の最新の手前訂正、048の修正接触範囲、050/051の同一点対応、保護対象ハッシュを維持した。これらはデータ処理の整合性検査で、追跡精度の独立評価ではない。

## 研究上の判断

得点側が両方ある入力表まで進められたことには実務上の価値がある。一方、8候補・1試合で配置の優劣や失点理由を論じる段階ではない。サーブ側や選手・スコアなどの交絡があり、配置と得点の関連があっても原因とは言えない。モデルを回して数値だけ増やすことは今回は行わない。

次は、この入力規則を保ったまま、残る6候補の開始と入力末尾が同じプレー中に入るかを既存の連続映像で点検する。正確な終端1フレームの追及は、序盤入力の確認には必須としない。その後に、勝敗や追跡の良否で選ばない別試合の標本を増やす。球追加との公平な比較は、今は球が整備された2例とも手前得点のため成立しない。

## 出力

- `requested_player_observations.csv`: 4選手×31要求時点×8候補。欠測と出力開始前を区別。
- `prefix_frame_features.csv`: 248時点のペア配置特徴。
- `snapshot_player_features.csv` / `prefix_player_summary_features.csv`: 8行の1時点版／短区間版。ラベルは別ファイル。
- `human_outcomes.csv`: 既存の最新人手得点側と確認記録ID。
- `rally_metadata.csv`: 開始根拠・群・時間範囲・観測数・素材由来。モデル入力にしない。
- `contact_timing_sensitivity.csv`: AI接触候補の範囲内での特徴変動。独立93ラリーとは数えない。
- `protocol.json` / `result.json` / `verification.json`: 固定条件・集計・整合性検査。

再実行用は `outputs/ball_quality_20261103/build_player_prefix_expansion.py`、独立再計算は `verify_player_prefix_expansion.py`、図と本文は `report_player_prefix_expansion.py`。元の追跡・注釈・正規参照は更新していない。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print('Wrote observation_availability.png and README.md')

if __name__=='__main__':main()
