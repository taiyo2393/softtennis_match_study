"""本人の参照CSVを比較。標準ライブラリのみ。--self-test は一時領域だけを使う。"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MOVING = {"single", "compound", "motion"}
KNOWN = {"none", "single", "compound", "motion"}
DIR3 = {"front", "lateral", "back"}
DIR8 = {"straight_front", "right_front", "right_lateral", "right_back",
        "straight_back", "left_back", "left_lateral", "left_front"}
FIELDS = ["id", "point", "contact_frame", "front_side", "motion", "unknown_reason",
          "screen_dir_final", "screen_dir_prelim", "player_dir8_final", "player_dir3_final",
          "player_dir8_prelim", "feint", "confidence", "note", "seconds", "annotator", "saved_at"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def auto_rows(path, method=None):
    rows = [r for r in read_csv(path) if math.isclose(float(r["W_sec"]), .4)
            and math.isclose(float(r["D_m"]), .3)
            and (method is None or r["coordinate_method"] == method)]
    indexed = {r["id"]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"主条件のIDが重複: {path}")
    return indexed


def load_sources():
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    items = {i["id"]: i for i in data["items"]}
    v1 = auto_rows(BASE / "net_motion/motions.csv")
    v2 = BASE / "net_motion/v2/motions.csv"
    sources = {"v1": v1, "v2 箱": auto_rows(v2, "bbox"), "v2 足首": auto_rows(v2, "ankle")}
    claude = {}
    for r in read_csv(BASE / "net_motion/review_claude_v1.csv"):
        # Claudeの目視対象はv1のslot。wrong_playerを新対象の動きへ読み替えない。
        old = v1[r["id"]]
        claude[r["id"]] = dict(classification=(r["claude_single_compound"] or "motion")
                               if r["claude_motion"] == "motion" else r["claude_motion"],
                               final_direction3=r["claude_direction3"], front_slot=old["front_slot"],
                               front_side=old["front_side"], issue=r["claude_issue"])
    sources["Claude（参考）"] = claude
    return items, sources


def load_labels(path, items):
    rows = read_csv(path)
    if not rows:
        raise ValueError("CSVにデータ行がありません")
    missing = set(FIELDS) - rows[0].keys()
    if missing:
        raise ValueError(f"必要な列がありません: {sorted(missing)}")
    indexed = {}
    for line, r in enumerate(rows, 2):
        ident = r["id"]
        if ident not in items:
            raise ValueError(f"{line}行: 対象外ID {ident}")
        if ident in indexed:
            raise ValueError(f"{line}行: ID重複 {ident}")
        item = items[ident]
        if (int(r["point"]) != item["point"] or int(r["contact_frame"]) != item["contact_frame"]
                or r["front_side"] != item["front_side"]):
            raise ValueError(f"{line}行: 打球のメタデータ不一致 {ident}")
        if r["motion"] not in {"", "none", "single", "compound", "unknown"}:
            raise ValueError(f"{line}行: 不正なmotion {r['motion']}")
        if r["motion"] == "unknown" and r["unknown_reason"] not in {"not_visible", "wrong_player", "other"}:
            raise ValueError(f"{line}行: unknownの理由がありません")
        if r["player_dir3_final"] and r["player_dir3_final"] not in DIR3:
            raise ValueError(f"{line}行: 不正な3方向")
        if r["player_dir8_final"] and r["player_dir8_final"] not in DIR8:
            raise ValueError(f"{line}行: 不正な8方向")
        if not math.isfinite(float(r["seconds"])) or float(r["seconds"]) < 0:
            raise ValueError(f"{line}行: 作業時間が不正")
        indexed[ident] = r
    return indexed


def agreement(labels, predictions, items):
    result = {"presence": [0, 0], "direction3": [0, 0], "complexity": [0, 0]}
    counts = dict(shared=0, human_unknown=0, human_blank=0, role_unknown=0,
                  person_changed=0, other_unknown=0, auto_unknown=0,
                  direction_missing=0, complexity_missing=0)
    for ident, h in labels.items():
        if ident not in predictions:
            continue
        counts["shared"] += 1
        if h["motion"] in {"", "unknown"}:
            counts["human_blank" if not h["motion"] else "human_unknown"] += 1
            continue
        p, item = predictions[ident], items[ident]
        if p.get("issue") and "wrong_player" in p["issue"]:
            counts["other_unknown"] += 1
            continue
        if not item["front_slot"] or not p["front_slot"]:
            counts["role_unknown"] += 1
            continue
        if (item["front_slot"], item["front_side"]) != (p["front_slot"], p["front_side"]):
            counts["person_changed"] += 1
            continue
        classification = p["classification"]
        # Claudeの空欄は参照未記入。自動のunclassifiableは棄却予測として不一致。
        if classification == "":
            counts["other_unknown"] += 1
            continue
        if classification not in KNOWN:
            counts["auto_unknown"] += 1
        result["presence"][1] += 1
        result["presence"][0] += int(classification in KNOWN and
                                      (h["motion"] in MOVING) == (classification in MOVING))
        if h["motion"] in MOVING and classification in MOVING:
            if h["player_dir3_final"] not in DIR3:
                counts["direction_missing"] += 1
            else:
                result["direction3"][1] += 1
                result["direction3"][0] += int(h["player_dir3_final"] == p["final_direction3"])
        if h["motion"] in MOVING:
            if classification == "motion":
                # Claudeが有無だけを付けて単・複を付けていない場合。
                counts["complexity_missing"] += 1
            else:
                result["complexity"][1] += 1
                result["complexity"][0] += int(h["motion"] == classification)
    return result, counts


def rate(pair):
    match, n = pair
    return f"{100 * match / n:.1f}% ({match}/{n})" if n else "— (0/0)"


def report(labels, items, sources, stream=None):
    out = stream or sys.stdout
    print("本人の目視ラベルとの一致（W=0.4秒・D=0.3m）", file=out)
    print(f"入力={len(labels)}件 / 未入力={sum(not r['motion'] for r in labels.values())}件 / "
          f"本人の判定不能={sum(r['motion']=='unknown' for r in labels.values())}件", file=out)
    print("| 比較対象 | 動きの有無 | 3方向（双方が動きあり） | 単・複（本人が動きあり） |", file=out)
    print("|---|---:|---:|---:|", file=out)
    results = {}
    for name, predictions in sources.items():
        metrics, counts = agreement(labels, predictions, items)
        results[name] = (metrics, counts)
        print(f"| {name} | {rate(metrics['presence'])} | {rate(metrics['direction3'])} | {rate(metrics['complexity'])} |", file=out)
    print("\n件数監査（本人の判定不能・未入力を先に除き、対象選手の一致を確認）", file=out)
    print("| 比較対象 | 共通ID | 本人未入力 | 本人不能 | slot未確定 | 対象選手違い | 参照未判定 | 自動不能※ | 本人方向欠損 | 参照単複欠損 |", file=out)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|", file=out)
    keys = ["shared", "human_blank", "human_unknown", "role_unknown", "person_changed", "other_unknown", "auto_unknown", "direction_missing", "complexity_missing"]
    for name, (_, counts) in results.items():
        print("| " + name + " | " + " | ".join(str(counts[k]) for k in keys) + " |", file=out)
    print("\n※ 自動の判定不能は有無と単・複の不一致に含める。3方向は双方が動きありの件だけ。", file=out)
    print("単・複では自動の「なし」も不一致。方向は自動のfinal_direction3（複は最終区間）を使用。", file=out)
    print("本人の方向未入力は方向集計から除外し、上表に数を表示。途中保存のCSVも読めるが、完了後のCSVを主評価に使う。", file=out)
    print("Claudeは20件中、記入済み・同一対象の項目だけを比較。20件は非無作為でv2開発にも使用済み。", file=out)
    print("井箟ほか（1986）の書き写し一致度93%は参考値。今回の映像判定の合格基準・直接の比較値にはしない。", file=out)
    print("人の3方向は斜め前→前・斜め後ろ→後ろ。自動の連続変位の45°境界と完全には対応しない。", file=out)
    return results


class ComparisonTests(unittest.TestCase):
    def test_reference_unknown_and_missing_direction(self):
        items = {str(i): dict(front_slot="near_front", front_side="N") for i in range(4)}
        labels = {str(i): dict(motion="single", player_dir3_final="front") for i in range(4)}
        predictions = {str(i): dict(classification="single", final_direction3="front",
                                   front_slot="near_front", front_side="N") for i in range(4)}
        items["0"]["front_slot"] = ""
        predictions["1"].update(issue="wrong_player", classification="")
        predictions["2"].update(classification="motion", final_direction3="")
        labels["3"]["player_dir3_final"] = ""
        metrics, counts = agreement(labels, predictions, items)
        self.assertEqual(metrics, dict(presence=[2, 2], direction3=[0, 1], complexity=[1, 1]))
        self.assertEqual(counts["role_unknown"], 1)
        self.assertEqual(counts["other_unknown"], 1)
        self.assertEqual(counts["direction_missing"], 1)
        self.assertEqual(counts["complexity_missing"], 1)

    def test_denominators(self):
        items = {str(i): dict(front_slot="near_front", front_side="N") for i in range(7)}
        hs = ["single", "compound", "single", "none", "unknown", "single", "single"]
        ps = ["single", "none", "compound", "unclassifiable", "single", "single", "single"]
        labels = {str(i): dict(motion=m, player_dir3_final="front") for i, m in enumerate(hs)}
        predictions = {str(i): dict(classification=m, final_direction3="front" if i == 0 else "back",
                                   front_slot="near_front", front_side="N") for i, m in enumerate(ps)}
        predictions["5"]["front_slot"] = "near_rear"
        predictions["6"]["classification"] = ""  # 未判定の参照
        metrics, counts = agreement(labels, predictions, items)
        self.assertEqual(metrics, dict(presence=[2, 4], direction3=[1, 2], complexity=[1, 3]))
        self.assertEqual(counts["auto_unknown"], 1)
        self.assertEqual(counts["person_changed"], 1)
        self.assertEqual(counts["human_unknown"], 1)
        self.assertEqual(counts["other_unknown"], 1)

    def test_real_sources_and_temporary_csv(self):
        items, sources = load_sources()
        self.assertEqual(len(items), 30)
        self.assertEqual(len(sources["Claude（参考）"]), 20)
        rows = []
        for n, item in enumerate(items.values()):
            r = {k: "" for k in FIELDS}
            r.update({k: item[k] for k in ("id", "point", "contact_frame", "front_side")})
            r.update(motion=["none", "single", "compound", "unknown"][n % 4], seconds="12.5",
                     annotator="ダミー", note='引用符"、カンマ,\n改行', confidence="high", feint="unknown")
            if r["motion"] == "unknown":
                r["unknown_reason"] = "not_visible"
            elif r["motion"] in MOVING:
                r.update(screen_dir_final="up", player_dir8_final="straight_front" if item["front_side"] == "N" else "straight_back",
                         player_dir3_final="front" if item["front_side"] == "N" else "back")
                if r["motion"] == "compound":
                    r.update(screen_dir_prelim="down", player_dir8_prelim="straight_back" if item["front_side"] == "N" else "straight_front")
            rows.append(r)
        with tempfile.TemporaryDirectory(prefix="motion-labels-selftest-") as directory:
            path = Path(directory) / "motion_labels.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            labels = load_labels(path, items)
            self.assertEqual(len(labels), 30)
            self.assertEqual(next(iter(labels.values()))["note"], rows[0]["note"])
            results = report(labels, items, sources, io.StringIO())
            self.assertEqual(results["Claude（参考）"][1]["shared"], 20)
            proc = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(path)],
                                  capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("v2 足首", proc.stdout)
            # 重複は黙って最終行を採用しない。
            with path.open("a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, FIELDS).writerow(rows[0])
            with self.assertRaisesRegex(ValueError, "ID重複"):
                load_labels(path, items)
        self.assertFalse(path.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="?", type=Path, help="本人の motion_labels.csv のパス")
    parser.add_argument("--self-test", action="store_true", help="ダミーCSVを一時領域で生成・比較・削除")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.labels:
        parser.error("motion_labels.csv のパス、または --self-test が必要です")
    try:
        items, sources = load_sources()
        report(load_labels(args.labels, items), items, sources)
    except (OSError, ValueError, KeyError) as exc:
        print(f"比較失敗: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
