"""Build the shot table from human labels only; reuse original coordinate helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PIPE, ROOT, module, labels, ignore_large, md_table, dump, sha
import pandas as pd

OUT = Path(__file__).resolve().parent
FRAME_SEC = 1001 / 60000


def main():
    old = module('original_shot_table', PIPE / 'build_shot_table.py')
    inputs = old.Inputs()
    inventory = inputs.by_point(ROOT / 'outputs/match2_point_inventory_20261117/point_inventory.csv')
    tracking = inputs.by_point(PIPE / 'tracking/summary.csv')
    human = labels()
    rows = []
    for point, g in human.groupby('point'):
        index = old.position_index(inputs, int(point), tracking[int(point)])
        game, near, far = old.POINT_CONTEXT[int(point)]
        assert int(inventory[int(point)]['game']) == game
        for rally_no, rally in g.groupby('rally_no'):
            rally = rally.sort_values('frame').reset_index(drop=True)
            for i, h in rally.iterrows():
                r = {col: '' for col in old.LABEL_COLUMNS}
                r.update(point=int(point), rally=str(rally_no), rally_no=int(rally_no),
                         shot_index=i+1, contact_frame=int(h.frame), hitter_side=h.side,
                         is_serve=old_value_bool(h.is_serve), shot_outcome=h.outcome)
                r.update(game=game, near_pair=near, far_pair=far,
                         hitter_pair=near if h.side == 'N' else far,
                         point_winner=inventory[int(point)]['point_winner'])
                r['hitter_won_point'] = r['hitter_pair'] == r['point_winner']
                r['players_available'] = 0
                for slot, positions in index.items():
                    p = old.nearest_position(positions, int(h.frame))
                    r[f'{slot}_x_m'] = p[0] if p else None
                    r[f'{slot}_y_m'] = p[1] if p else None
                    r[f'{slot}_frame_offset'] = p[2] if p else None
                    r['players_available'] += p is not None
                r['near_positions_trusted'] = int(point) != 8
                r['far_positions_trusted'] = True
                for prefix, side in [('hitter', h.side), ('opponent', 'F' if h.side == 'N' else 'N')]:
                    lateral, depth = old.pair_metrics(r, side)
                    r[f'{prefix}_pair_lateral_m'], r[f'{prefix}_pair_depth_m'] = lateral, depth
                prev_gap = int(h.frame - rally.iloc[i-1].frame) if i else None
                next_gap = int(rally.iloc[i+1].frame - h.frame) if i+1 < len(rally) else None
                r.update(previous_shot_interval_frames=prev_gap, next_shot_interval_frames=next_gap,
                         previous_shot_interval_sec=prev_gap * FRAME_SEC if prev_gap else None,
                         next_shot_interval_sec=next_gap * FRAME_SEC if next_gap else None)
                rows.append(r)
    columns = old.LABEL_COLUMNS + old.ADDED_COLUMNS + ['rally_no', 'previous_shot_interval_frames',
        'previous_shot_interval_sec', 'next_shot_interval_frames', 'next_shot_interval_sec']
    df = pd.DataFrame(rows)[columns]
    assert len(df) == 32 and df.hitter_slot.eq('').all()
    assert df.groupby(['point','rally_no']).size().to_dict() == {(7,1):2,(7,2):6,(8,1):8,(9,1):8,(10,1):4,(11,1):4}
    gaps = df.previous_shot_interval_frames.dropna()
    assert len(gaps) == 26 and gaps.min() > 0
    p8 = df[df.point == 8]
    assert p8.near_positions_trusted.eq(False).all()
    assert p8.loc[p8.hitter_side == 'N', 'hitter_pair_lateral_m'].isna().all()
    df.to_csv(OUT / 'shot_table.csv', index=False, float_format='%.10g')
    summary = df.groupby('point').agg(shots=('point','size'), four_players=('players_available',lambda x:int((x==4).sum())))
    summary.to_csv(OUT / 'point_summary.csv')
    readme = f'''# 本人ラベルによる打球表

正解は `../../labels_human/human_labels_20261118.csv` の condition=without_candidates の32件だけ。labels/ は読まない。
元の `../../build_shot_table.py` の LABEL_COLUMNS と ADDED_COLUMNS を同じ順序・定義で保持し、末尾に rally_no と前後打球間隔を追加。
point/frame/side/is_serve/outcome を point/contact_frame/hitter_side/is_serve/shot_outcome に対応させた。
rally は本人の rally_no の文字列表現、shot_index は同じ点・rally_no 内の時刻順連番。hitter_slot・confidence・tracking_issue・note は本人が付けていないため空欄。
第7点の2つの流れを分離。第9点の同一rally_no内の2サーブ、第11点の最初の非サーブ印を変更しない。

座標は元スクリプトの position_index / nearest_position / pair_metrics を import して使用。
正規化コート座標 x×10.97、y×23.77 m、ネットy=11.885 m。接触フレーム±3F内の最近観測、同距離は早い方。補間・外挿・範囲内へのクリップなし。
全4人のraw座標は保持。第8点手前は線審追跡のため near_positions_trusted=False とし、そのペアの横幅・前後差を空欄にする。
trusted=True はこの既知の除外に該当しない意味で、追跡精度の保証ではない。game/pair/point_winner等の文脈は元と同じ。

間隔は同じpoint・rally_no内だけ。前後がない場合は空欄。秒換算=F×1001/60000。
間隔{len(gaps)}件、中央値{gaps.median():g}F（{gaps.median()*FRAME_SEC:.3f}秒）、最短{gaps.min():g}F（{gaps.min()*FRAME_SEC:.3f}秒）、最長{gaps.max():g}F。
第9点の2本のサーブ間も本人の同じrally_noに従い含む。

{md_table(summary.reset_index())}

実行: `study\\.venv\\Scripts\\python.exe -B -X utf8 outputs/shot_pipeline_20261117/v_human/shot_table/build_shot_table.py`
'''
    (OUT / 'README.md').write_text(readme, encoding='utf-8')
    inputs.assert_unchanged()
    dump(OUT / 'input_manifest.json', {**inputs.hashes,
         'human_labels_sha256': sha(PIPE / 'labels_human/human_labels_20261118.csv'),
         'original_script_sha256': sha(old.__file__)})
    ignore_large()
    print(summary.to_string())
    print(f'32 shots; intervals={len(gaps)}, median={gaps.median()}F, minimum={gaps.min()}F')


def old_value_bool(v):
    return str(v).lower() == 'true'


if __name__ == '__main__':
    main()
