"""AI diagnostic observations; source labels and adopted predictions remain unchanged."""
from collections import Counter
import csv
import hashlib
import json

from build_joint_failure_review import COMPARE,OUT,SOURCE,getcases

OBSERVATIONS={
1:('preparation_or_between_points','手元のボールを保持する動作。既存注記もサービス前の保持。','high'),
2:('preparation_or_between_points','サーブ準備でボールを床につき、手元に戻す連続動作。','high'),
3:('moving_ball_or_stroke_context','奥側選手が低い位置で打球する場面。投影位置が身体・ラケットの近くで、遮蔽とブレの寄与は未確定。','medium'),
4:('moving_ball_or_stroke_context','奥側選手の打球動作。頭上付近のボールに縦方向の像の伸びが見える。','high'),
5:('moving_ball_or_stroke_context','奥側選手の打球動作付近。ボールの身体との重なり・ブレが候補。対象時刻の可視性は独立確認が必要。','medium'),
6:('preparation_or_between_points','近接撮影でサーブ前の腕上げ動作。既存注記は保持。保持からトスへの境界は未確定。','medium'),
7:('moving_ball_or_stroke_context','打球中で手前と奥の選手が投影上重なる。遮蔽候補だが対象時刻の可視性は断定しない。','medium'),
8:('shared_white_glove_false_positive','両モデルの推定位置が線審の白い手袋に一致。既存注釈位置付近には別のボール像があり、前後フレームで移動する。','high'),
9:('preparation_or_between_points','近接撮影でボールを手元に保持。','high'),
10:('preparation_or_between_points','選手間でボールを受け渡す動作。既存注記の「ついている」は動作の記述と一致しない可能性。','high'),
11:('preparation_or_between_points','ボールを手元に保持しながらラケットを動かす準備動作。','high'),
12:('moving_ball_or_stroke_context','奥側選手が打球する場面。ボールの像の伸び・身体との重なりが候補。','medium'),
13:('moving_ball_or_stroke_context','打球・ボール移動のある場面。小さい像とブレの候補で、単一原因は未確定。','medium'),
14:('moving_ball_or_stroke_context','前後フレームでボールが移動するが、対象時刻の像を十分に判別できない。可視性・位置注釈の確認候補。','low'),
15:('moving_ball_or_stroke_context','手前選手と奥側の打球位置が投影上近い。身体との重なりが候補。','medium'),
16:('moving_ball_or_stroke_context','打球動作のある場面。身体・背景との重なりとブレの寄与は未確定。','medium'),
17:('moving_ball_or_stroke_context','近い選手が打球し、注釈位置が身体・腕付近にある。遮蔽とブレの候補。','medium'),
18:('preparation_or_between_points','選手の停止・歩行が主体のポイント間に見える。既存注記はサービス前。正確な開始終了は未確認。','medium'),
19:('preparation_or_between_points','選手の停止・歩行が主体。既存注記はボール保持。画像だけで保持位置は確定しない。','medium'),
20:('preparation_or_between_points','サーブ準備に見える停止動作。既存注記は手での保持。','medium'),
21:('moving_ball_or_stroke_context','近接した打球動作とラケットの強い像の伸び。ボールの身体との重なりも候補。','medium'),
22:('moving_ball_or_stroke_context','ボール移動と選手の構えのある場面。身体との重なり・小さい像の候補。対象時刻の可視性は未確定。','medium'),
23:('shared_white_glove_false_positive','両モデルの推定位置が線審の上げた白い手袋に一致。既存注釈位置付近には別のボール像がある。','high'),
24:('moving_ball_or_stroke_context','近接した打球動作。身体・ラケットとの重なりの候補で、ボール中心の直接視認は未確定。','medium'),
25:('moving_ball_or_stroke_context','打球・ボール移動のある場面。身体との重なりが候補。','medium'),
26:('moving_ball_or_stroke_context','打球中で前後の選手が投影上近い。遮蔽と小さい像の候補。','medium'),
27:('moving_ball_or_stroke_context','近い選手の打球後、ボール像が速く移動し長く伸びる。ブレによる見落とし候補。','high'),
}


def main():
    cases=getcases();assert len(cases)==len(OBSERVATIONS)==27
    rows=[]
    for i,c in enumerate(cases,1):
        context,notes,confidence=OBSERVATIONS[i]
        rows.append(dict(case_id=c['case_id'],candidate_id=c['candidate_id'],local_frame=c['local_frame'],
            source_timestamp_sec=c['source_timestamp_sec'],diagnostic_context=context,
            ai_diagnostic_confidence=confidence,ai_visual_observation=notes,existing_notes=c['existing_notes'],
            independent_human_review='not_performed_in_this_step',source_label_changed=False,
            contact_sheet=f'temporal_page_{(i-1)//3+1:02d}.jpg',
            inspected_local_frames=','.join(str(max(0,min(c['frame_count']-1,int(c['local_frame'])+d))) for d in (-12,-6,0,6,12))))
    with (OUT/'triage.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts=dict(Counter(r['diagnostic_context'] for r in rows))
    label=SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv'
    current_sha=hashlib.sha256(label.read_bytes()).hexdigest()
    expected=json.loads((COMPARE/'protocol.json').read_text())['label_sha256']
    assert current_sha==expected,'Source labels changed since comparison; review provenance before continuing'
    summary=dict(context_counts=counts,review_kind='AI diagnostic from sampled temporal frames, not an independent annotation or full video playback review',
        inspection='All 27 targets at offsets -12,-6,0,+6,+12; two glove cases also at -2,0,+2 with raw crops and 3x pixel enlargement.',
        source_labels_sha256=current_sha,source_labels_unchanged_since_comparison=True,
        score_recomputed=False,adopted_predictions_changed=False,
        evidence='Two glove cases: original clip decoded sequentially, cache exactly equals JPEG quality95 reencoding of corresponding frame.',
        scope_warning='Counts are descriptive of 27 selected joint failures only, not prevalence in all rallies or an eligible-rally count.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    text='''# ボール追跡：両方式が失敗した27標本の点検

既存注釈と両モデルの推定を照合し、27標本の前後フレームをAIが点検した。
独立した正解注釈の確定や、27件の原因が全て確定したという意味ではない。

| 暫定的な場面区分 | 標本数 |
|---|---:|
| サーブ準備・ポイント間の候補 | 9 |
| 打球動作・ボール移動のある場面（原因未確定を含む） | 16 |
| 両モデルが線審の白い手袋を検出 | 2 |

この内訳は失敗標本だけの診断であり、全ラリーでの発生率ではない。
サーブ準備区分には保持・バウンド・受け渡し・トス前後の候補を含み、全てが保持ボールとは限らない。
打球中の候補では身体との重なり、ブレ、小さい像などが見られるが、各要因の寄与は確定していない。
単一フレームで球が見えない場合に、前後の軌道から推定した位置を「観測された正解」として補わない。

## 明確な改善対象：線審の白い手袋

- 013の435フレーム：現行版(633,540)、TOTNet(633.75,543.75)。両者は線審の白い手袋付近を示す。既存注釈は(605,606)で、別のボール像の位置付近。
- 018の216フレーム：現行版(648,534)、TOTNet(645,536.25)。こちらも上げた白い手袋付近。既存注釈は(740,575)で、別のボール像の位置付近。

二つのモデルの一致を正解の根拠にできない例である。
どちらも近くに編集の切り替えがあるが、対象フレームは元の区間動画を先頭から復号し、
同じ時刻の画像をJPEG品質95で再圧縮した結果とキャッシュが画素単位で一致した。
この2件はキャッシュのフレームずれが原因ではない。注釈の座標や可視性は変更していない。

## 評価と今後の扱い

現行版75.0%、TOTNet64.8%という既存300標本の比較値はそのまま残す。
「見えるボールを全て検出する」評価には、手で持つ球やポイント終了後の球も含まれる。
一方、ラリー中の配置・打球を分析する用途では、プレー中の球の検出を別に測る必要がある。
場面別の評価には、失敗例だけでなく成功例も含めた全標本への一貫した区分が必要になる。
今回の9件だけを除いて改善後の精度として報告しない。

次の改善候補は、白い手袋のような球以外の物体への誤検出への対処と、打球中のブレ・遮蔽に対する処理を分けること。
選手の手元の球や、人物と重なる本物の球を消す恐れがあるため、人物領域の一括除外は適さない。
この27件は既に見たテスト由来なので、これを使って開発した変更の最終性能は別の未使用区間で評価する。

## 確認用資料

- triage.csv：27件の観察メモ。AIの診断候補であり独立人手注釈ではない。
- temporal_page_01.jpg〜09.jpg：対象時刻±12フレーム、6フレーム間隔の等倍切り出し。オレンジ枠が対象時刻。
- 08_013_f435_coordinate_conflict.png / 23_018_f216_coordinate_conflict.png：生画像と位置の照合。上段は無印、下段は対象時刻にのみ印。
- *_raw_zoom3.png：この2件を最近傍法で3倍表示。元の細部を補う画像生成は使用していない。
- videos/：準備動作、打球中のブレ、白手袋の2件の計4本。通常速度と0.25倍速を収録し、対象時刻を短く停止。
- 動画の緑十字は既存注釈、水色丸は現行版、桃色丸はTOTNet。モデルの印は評価対象の1フレームにのみ描き、未推論時刻のTOTNet軌跡は描かない。
- video_render_report.json：4本全フレームの復号とフレーム数を検査した記録。動画を作成・復号できることは注釈の正しさの証明ではない。
- coordinate_conflict_report.json：2件の元動画・キャッシュ照合。
- summary.json：区分数と元注釈ファイルの不変確認。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
