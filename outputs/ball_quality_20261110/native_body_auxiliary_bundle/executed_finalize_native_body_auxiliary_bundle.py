import sys,json,unittest,subprocess,shutil
from collections import Counter
from pathlib import Path
from build_native_body_auxiliary_bundle import OUT,NATIVE,BODY,resolve,read,load,sha,dump,write,ROOT


class MergeSafety(unittest.TestCase):
    def native(self):return dict(x1='0',y1='0',x2='10',y2='10',bottom_center_x='5',bottom_center_y='10',track_id='9')
    def body(self):return dict(body_ID_candidate='1',body_box_json='[0,0,10,10]')
    def test_native_coordinates_survive_conflicting_body(self):
        r=resolve(self.native(),self.body(),{'other':self.native()})
        self.assertEqual((r['analysis_x'],r['analysis_y'],r['native_raw_id']),('5','10','9'))
        self.assertEqual(r['evidence_state'],'native_box_candidate')
    def test_body_never_creates_foot_coordinates(self):
        r=resolve(None,self.body(),{})
        self.assertEqual(r['identity_candidate_present'],1)
        self.assertEqual((r['analysis_x'],r['analysis_y']),('',''))
    def test_duplicate_of_existing_person_is_withheld(self):
        r=resolve(None,self.body(),{'near_front':self.native()})
        self.assertEqual(r['identity_candidate_present'],0)
        self.assertEqual(r['evidence_state'],'unsupported_body_conflict')
        self.assertEqual(r['body_candidate_available'],1)
    def test_no_evidence_remains_missing(self):
        r=resolve(None,None,{})
        self.assertEqual((r['identity_candidate_present'],r['native_position_proxy_present']),(0,0))


def finalize():
    p=load(OUT/'protocol.json');rows=read(OUT/'combined_evidence_states.csv');frames=read(OUT/'frame_availability_not_accuracy.csv');pairs=read(OUT/'native_pair_geometry.csv')
    assert len(rows)==2480 and len(frames)==620 and len(pairs)==1240
    assert len({(r['tag'],r['policy'],r['source_frame'],r['player_slot']) for r in rows})==2480
    assert all(r['analysis_x']==r['analysis_y']=='' for r in rows if r['native_position_proxy_present']=='0')
    assert all(r['native_pair_distance_image_px']=='' for r in pairs if r['native_pair_geometry_available']=='0')
    summaries=[]
    for mode in p['modes']:
        group=[r for r in rows if r['policy']==mode];fr=[r for r in frames if r['policy']==mode]
        summaries.append(dict(policy=mode,states=dict(Counter(r['evidence_state'] for r in group)),
            four_native_position_times=sum(r['four_native_position_proxies']=='1' for r in fr),
            four_identity_candidate_times=sum(r['four_identity_candidates']=='1' for r in fr),
            independent_accuracy_established=False))
    tests=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)),'test'],capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    dump(OUT/'verification.json',dict(status='passed',merged_slot_rows=2480,
        all_missing_positions_preserved=True,geometry_never_created_from_body=True,
        merge_invariant_tests=4,test_output=tests.stdout+tests.stderr,
        summaries=summaries,protected_inputs_unchanged=True))
    dump(OUT/'recommended_video_metadata.json',dict(status='CLI_metadata_unavailable_sandbox_DNS',
        url='https://www.youtube.com/watch?v=iRRLb6MLtXc',
        title='2024年 全日本ソフトテニス選手権大会 男子 決勝 内本隆文・内田理久 対 船水颯人・上松俊貴',
        publisher='公益財団法人日本ソフトテニス連盟',
        evidence='Official YouTube title/search and browser playback at approximately5:14; UI quality menu lists1080p60 maximum. All four court players and score overlay observed in sampled wide view.',
        full_video_quality_or_effective_motion_fps_verified=False,
        separate_match_and_all_four_players_differ_from_current_2025_final=True,
        downloaded_video_file=False))
    (OUT/'README.md').write_text('''# 既存の座標と身体だけの補助候補を併用する開発用データ

全10区間×31時点×4スロット×2方式の2480行を固定した。既存の検出枠・スロット・raw ID・画像上の足元代理点を優先し、書き換えない。既存の枠がない時点だけ、身体ID候補を補助状態として残す。別の既存スロットとIoU0.5以上で重なる身体候補は競合として残し、補助IDを採用しない。

006は既存の手前先行初期化候補と、奥側の身体候補29件を併用できる。ただし身体のみの状態には座標がない。4人の位置代理点がそろうのは従来どおり11/31時点。身体候補を含めて4スロットを支持するのは20/31時点であり、4人の位置や本人同定の正解が20件できたという意味ではない。

他区間で身体経路の欠測が増えても既存の出力を消さない構成にした。両方式を一律に保存し、区間ごとに好都合な方式へ切り替えていない。native_pair_geometry.csvは両者に既存座標がある場合のみ画像上のペア間距離を持つ。身体候補から座標や距離を作った行は0。

combined_evidence_states.csvのnative_box_candidateも独立確認済みの人物・接地点ではない。body_only_identity_hypothesisはオフライン身体対応の候補、unsupportedは欠測。候補の支持数を追跡精度として使用しない。006の補助候補は後の初期化画像を使う。current22、得点側・区間ラベル、公開済み評価、元の追跡ファイルは不変。正本に置き換えたデータではない。

## 別試合での次の評価

ユーザーは現在別試合を保存していないと回答した。推奨候補は日本ソフトテニス連盟公式の2024年男子決勝（内本隆文・内田理久 対 船水颯人・上松俊貴）。現在の2025年男子決勝とは全4選手が異なる。約5分の再生画面でコート全体と4人、得点表示を確認し、画質メニューで1080p60を確認した。全編を監査したものではない。CLIでのメタデータ取得はDNS解決できず、取得成功を主張しない。

https://www.youtube.com/watch?v=iRRLb6MLtXc

別試合の結果を見る前に、対象のラリー・比較する凍結済み方式・必要な解像度対応とコート校正・参照の付け方を固定する。失敗を見て修正した場合、その動画は開発データになり最終的な独立評価と区別する。同会場の別試合は、別会場・一般選手全体への汎化を証明しない。表示上60fpsでも動きの実更新頻度はダウンロード後に別途調べる。

## ローカル映像の確認範囲

研究用videos.csvの登録はmatch1の1試合。rawには2ファイルあるが、編集版と元動画などの違いだけでは独立した別試合にならない。dataset/videosの218動画の代表3件は、人物のストローク撮影であり、4人の試合追跡の独立評価用とは確認できなかった。全218本やPC全体を調べたわけではない。ファイル名や数だけで別試合・別選手の評価数に算入していない。
''',encoding='utf-8')
    dump(OUT/'processing_decision.json',dict(status='frozen_development_auxiliary_bundle_not_canonical',
        next='User can download recommended distinct2024 final unchanged; inspect metadata and preregister held-out evaluation before processing. Existing match remains development data.',
        independent_match_available_locally=False,new_video_downloaded=False))
    for n in [ROOT/'build_native_body_auxiliary_bundle.py',Path(__file__)]:shutil.copy2(n,OUT/('executed_'+n.name))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    hashes={str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'}
    dump(OUT/'publication_hashes.json',hashes);assert all(sha(OUT/n)==h for n,h in hashes.items())
    print('Combined candidate bundle frozen;4 invariant checks passed; native positions and labels unchanged.',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='test':unittest.main(argv=[sys.argv[0]])
    else:finalize()
