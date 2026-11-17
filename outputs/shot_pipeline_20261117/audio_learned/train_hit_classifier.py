"""Small learned hit classifiers; all writes are confined to this directory.

No sklearn, pretrained weights, package installation, or retained WAV files.
Run from repo root with study/.venv/Scripts/python.exe -B -X utf8 <this file>.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, signal, special
from scipy.io import wavfile
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
PIPE = OUT.parent
FS, FRAME_SEC = 48000, 1001 / 60000
SEEDS = (17, 29, 43, 71, 101)
AUDIO = ['strength', 'half_width_ms', 'width_censored', 'low_rise_ln',
         'high_rise_ln', 'paired_rise_ln', 'low_peak_lag_within4_ms',
         'low_peak_lag_within20_ms', 'post50_energy_ratio', 'post50_drop_ln',
         'energy_concentration_20_100', 'spectral_flatness_100ms']
WRIST = ['wrist_peak_height', 'wrist_second_peak_height', 'wrist_peak_gap',
         'wrist_side_gap', 'n_players_with_peak', 'n_sides_with_peak']
METHODS = ['lr_audio', 'lr_audio_wrist', 'cnn', 'stack']
OLD = ['audio_v1', 'audio_v2', 'audio_motion', 'audio_wrist']
METRICS = ['recall_clear_f6', 'recall_clear_f10', 'recall_all_f10',
           'precision_proxy_f10', 'precision_1to1_f10', 'candidates_per_sec',
           'review_window_sum_ratio', 'review_window_union_ratio', 'f1_all_f10',
           'n_candidates', 'n_matched_clear_f10', 'n_matched_all_f10']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


def input_files():
    return [PIPE / p for p in ['audio/candidates.csv', 'audio/intervals.csv',
            'audio/candidate_features_v2.csv', 'pose/candidate_features.csv',
            'audiovisual/candidate_features.csv', 'audio/candidates_v2.csv',
            'pose/selected_candidates.csv', 'audiovisual/selected_candidates.csv',
            'tracking/point_07/protocol.json', 'audiovisual/combine_audio_video.py']] + [
            PIPE / f'labels/point_{p:02}.csv' for p in range(7, 12)]


class Dataset:
    def __init__(self):
        self.c = pd.read_csv(PIPE / 'audio/candidates.csv')
        self.ints = pd.read_csv(PIPE / 'audio/intervals.csv')
        self.labels = pd.concat([pd.read_csv(PIPE / f'labels/point_{p:02}.csv')
                                 for p in range(7, 12)], ignore_index=True)
        self.labels = self.labels[self.labels.confidence.isin(['clear', 'inferred'])].reset_index(drop=True)
        self.labels['label_id'] = [f'p{r.point:02}_{r.rally}_{r.shot_index:03}'
                                   for r in self.labels.itertuples()]
        self.audio = pd.read_csv(PIPE / 'audio/candidate_features_v2.csv')
        self.wrist = pd.read_csv(PIPE / 'pose/candidate_features.csv')
        self.motion = pd.read_csv(PIPE / 'audiovisual/candidate_features.csv')
        for table in (self.audio, self.wrist, self.motion):
            assert table.candidate_id.tolist() == self.c.candidate_id.tolist()
            assert (table[['point', 'rally', 'candidate_frame']].values ==
                    self.c[['point', 'rally', 'candidate_frame']].values).all()
        self.xa = self.audio[AUDIO].to_numpy(dtype=float)
        self.xw = np.column_stack([self.xa, self.wrist[WRIST].to_numpy(dtype=float)])
        self.points = self.c.point.to_numpy()
        self.n = len(self.c)
        self.clear = (self.labels.confidence == 'clear').to_numpy()
        self.lp = self.labels.point.to_numpy()
        self.nearest = np.full(self.n, np.inf)
        self.pairs = {}
        for tol in (6, 10):
            pairs = []
            for li, lab in self.labels.iterrows():
                for ci, cand in self.c.iterrows():
                    if lab.point != cand.point or lab.rally != cand.rally:
                        continue
                    delta = int(cand.candidate_frame - lab.contact_frame)
                    self.nearest[ci] = min(self.nearest[ci], abs(delta))
                    if abs(delta) <= tol:
                        pairs.append((abs(delta), int(lab.contact_frame), int(cand.candidate_frame),
                                      lab.label_id, cand.candidate_id, li, ci, delta))
            self.pairs[tol] = sorted(pairs)
        assert np.isfinite(self.nearest).all()

    def idx(self, points):
        return np.flatnonzero(np.isin(self.points, points))

    def match(self, selected, points, tol):
        ul, uc, rows = set(), set(), []
        for _, _, _, _, _, li, ci, delta in self.pairs[tol]:
            if selected[ci] and self.lp[li] in points and li not in ul and ci not in uc:
                ul.add(li)
                uc.add(ci)
                rows.append((li, ci, delta))
        return rows

    def metrics(self, selected, points, full=True):
        selected = np.asarray(selected, dtype=bool) & np.isin(self.points, points)
        lm = np.isin(self.lp, points)
        nclear, nlab, n = int((lm & self.clear).sum()), int(lm.sum()), int(selected.sum())
        m10 = self.match(selected, points, 10)
        nc10 = sum(self.clear[li] for li, _, _ in m10)
        row = dict(n_candidates=n, n_labels_clear=nclear, n_labels_all=nlab,
                   n_matched_clear_f10=int(nc10), n_matched_all_f10=len(m10),
                   recall_clear_f10=nc10 / nclear)
        if not full:
            return row
        m6 = self.match(selected, points, 6)
        duration = float(self.ints[self.ints.point.isin(points)].duration_sec.sum())
        summed, union = 0., 0.
        for seg in self.ints[self.ints.point.isin(points)].itertuples():
            frames = self.c.loc[selected & (self.points == seg.point) &
                                (self.c.rally == seg.rally), 'candidate_frame'].sort_values()
            end = float(seg.start_frame)
            for f in frames:
                lo, hi = max(seg.start_frame, f - 10), min(seg.end_frame, f + 10)
                summed += max(0., hi - lo)
                union += max(0., hi - max(end, lo))
                end = max(end, hi)
        row.update(duration_sec=duration, recall_clear_f6=sum(self.clear[li] for li, _, _ in m6) / nclear,
                   recall_all_f10=len(m10) / nlab,
                   precision_proxy_f10=float((selected & (self.nearest <= 10)).sum() / n) if n else 0.,
                   precision_1to1_f10=len(m10) / n if n else 0.,
                   candidates_per_sec=n / duration,
                   review_window_sum_ratio=summed * FRAME_SEC / duration,
                   review_window_union_ratio=union * FRAME_SEC / duration,
                   f1_all_f10=2 * len(m10) / (n + nlab))
        return row

    def choose(self, scores, points):
        idx = self.idx(points)
        # Exhaustive unique-score sweep: greedy clear matching is not monotone.
        thresholds = np.r_[np.unique(scores[idx]), np.nextafter(max(scores[idx]), np.inf)]
        trials = []
        for threshold in thresholds:
            row = self.metrics(scores >= threshold, points, False)
            row['threshold'] = float(threshold)
            row['feasible'] = bool(row['recall_clear_f10'] >= .95 - 1e-12)
            trials.append(row)
        feasible = [r for r in trials if r['feasible']]
        if feasible:
            best = min(feasible, key=lambda r: (r['n_candidates'], -r['threshold']))
        else:
            best = min(trials, key=lambda r: (-r['recall_clear_f10'], r['n_candidates'], -r['threshold']))
        return best


def spectrogram_cache(ds, ffmpeg):
    video = Path(json.loads((PIPE / 'tracking/point_07/protocol.json').read_text(encoding='utf-8'))['source_video'])
    key = dict(input_sha256={str(p.relative_to(PIPE)): sha(p) for p in input_files()[:2]},
               video=str(video), bytes=video.stat().st_size, mtime=video.stat().st_mtime_ns,
               sr=FS, n_fft=1024, hop_samples=96, mel_bands=64, hz=[100, 16000],
               window_ms=[-50, 150], shifts_ms=[0, -5, -2, 2, 5], version=1)
    cache = OUT / 'logmel_cache.npy'
    manifest = OUT / 'logmel_cache.json'
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) == key:
        return np.load(cache), key, True
    to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = 700 * (10 ** (np.linspace(to_mel(100), to_mel(16000), 66) / 2595) - 1)
    freqs = np.fft.rfftfreq(1024, 1 / FS)
    bank = np.maximum(0, np.minimum((freqs[None, :] - edges[:-2, None]) /
                      (edges[1:-1] - edges[:-2])[:, None],
                      (edges[2:, None] - freqs[None, :]) / (edges[2:] - edges[1:-1])[:, None]))
    bank /= np.maximum(bank.sum(axis=1, keepdims=True), 1e-12)
    images = np.empty((ds.n, 5, 64, 101), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix='tennis_learned_') as scratch:
        for si, seg in enumerate(ds.ints.itertuples()):
            start, end = seg.start_frame * FRAME_SEC - .3, seg.end_frame * FRAME_SEC + .3
            wav = Path(scratch) / f'interval_{si}.wav'
            subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y',
                            '-ss', f'{start:.9f}', '-i', str(video), '-t', f'{end-start:.9f}',
                            '-vn', '-ac', '1', '-ar', str(FS), '-c:a', 'pcm_s16le', str(wav)], check=True)
            sr, audio = wavfile.read(wav)
            assert sr == FS
            audio = audio.astype(np.float32) / 32768
            indices = np.flatnonzero((ds.points == seg.point) & (ds.c.rally == seg.rally))
            for i in indices:
                for j, shift in enumerate([0, -.005, -.002, .002, .005]):
                    centers = np.rint((ds.c.iloc[i].candidate_time_sec - start + shift +
                                       np.linspace(-.05, .15, 101)) * FS).astype(int)
                    samples = centers[:, None] + np.arange(-512, 512)[None, :]
                    assert samples.min() >= 0 and samples.max() < len(audio)
                    power = abs(np.fft.rfft(audio[samples] * signal.windows.hann(1024, sym=False), axis=1)) ** 2
                    images[i, j] = np.log(np.maximum(bank @ power.T, 1e-10))
    assert not Path(scratch).exists()
    np.save(cache, images)
    write_json('logmel_cache.json', key)
    return images, key, False


def logistic(x, y, train, l2=.1):
    # Imputation, missing indicators, mean and SD are fitted only on train.
    finite = np.isfinite(x)
    med = np.array([np.median(x[train, j][finite[train, j]]) if finite[train, j].any() else 0.
                    for j in range(x.shape[1])])
    clean = np.where(finite, x, med)
    clean = np.column_stack([clean, (~finite).astype(float)])
    mean, sd = clean[train].mean(0), clean[train].std(0)
    sd = np.where(sd < 1e-8, 1., sd)
    z = np.column_stack([np.ones(len(x)), (clean - mean) / sd])
    xt, yt = z[train], y[train].astype(float)

    def objective(w):
        logits = xt @ w
        loss = np.mean(np.logaddexp(0, logits) - yt * logits) + .5 * l2 * (w[1:] @ w[1:])
        grad = xt.T @ (special.expit(logits) - yt) / len(train)
        grad[1:] += l2 * w[1:]
        return loss, grad

    result = optimize.minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                               options={'maxiter': 400, 'gtol': 1e-8, 'ftol': 1e-12})
    assert result.success, result.message
    return special.expit(z @ result.x)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
                                 nn.AvgPool2d((8, 8)), nn.Flatten(),
                                 nn.Linear(144, 24), nn.ReLU(), nn.Dropout(.2), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class Learner:
    def __init__(self, ds, images, radius, device, epochs):
        self.ds, self.images, self.radius, self.device, self.epochs = ds, images, radius, device, epochs
        self.y = (ds.nearest <= radius).astype(float)
        self.base_cache, self.stack_cache = {}, {}
        self.fit_log, self.split_log = [], []

    def splits(self, train):
        points = sorted(set(self.ds.points[train]))
        if len(points) > 1:
            return [(train[self.ds.points[train] != p], train[self.ds.points[train] == p], 'point') for p in points]
        # Only deepest nesting in A has one training point. Block within point,
        # purge 0.4 s of neighbours: longer than image/feature receptive fields.
        result = []
        for valid in np.array_split(train, 3):
            fit = np.setdiff1d(train, valid)
            keep = []
            for i in fit:
                near = self.ds.c.iloc[valid]
                near = near[near.rally == self.ds.c.iloc[i].rally]
                keep.append(not (abs(near.candidate_time_sec - self.ds.c.iloc[i].candidate_time_sec) <= .4).any())
            fit = fit[np.asarray(keep)]
            assert len(fit) >= 8
            result.append((fit, valid, 'single_point_purged_time_blocks'))
        return result

    def base(self, train, seed):
        key = (tuple(train), seed)
        if key in self.base_cache:
            return self.base_cache[key]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Per mel-band normalization fitted on original training candidates only.
        original = self.images[train, 0]
        mean = original.mean(axis=(0, 2), keepdims=True)
        sd = np.maximum(original.std(axis=(0, 2), keepdims=True), .1)
        rng = np.random.default_rng(seed)
        cnn = TinyCNN().to(self.device)
        optimizer = torch.optim.AdamW(cnn.parameters(), lr=.003, weight_decay=.01)
        target = torch.tensor(self.y[train], dtype=torch.float32, device=self.device)
        xvalid = torch.tensor(((self.images[:, 0] - mean) / sd)[:, None], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        cnn.train()
        for epoch in range(self.epochs):
            variants = np.zeros(len(train), dtype=int)
            positive = self.y[train] == 1
            variants[positive] = rng.integers(0, 5, positive.sum())
            batch = self.images[train, variants].copy()
            # +/-1 dB gain in log power; only training positives augmented.
            batch[positive] += rng.uniform(-1, 1, positive.sum())[:, None, None] * np.log(10) / 10
            xt = torch.tensor(((batch - mean) / sd)[:, None], device=self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(cnn(xt), target)
            if epoch == 0:
                initial_loss = float(loss.detach().cpu())
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        cnn.eval()
        with torch.no_grad():
            pred = torch.sigmoid(cnn(xvalid)).cpu().numpy().astype(float)
        result = np.column_stack([logistic(self.ds.xa, self.y, train),
                                  logistic(self.ds.xw, self.y, train), pred])
        self.fit_log.append(dict(target_radius=self.radius, seed=seed, n_train=len(train),
                                 train_indices=list(map(int, train)), train_points=sorted(map(int, set(self.ds.points[train]))),
                                 initial_loss=initial_loss, final_loss=float(loss.detach().cpu()), epochs=self.epochs))
        self.base_cache[key] = result
        return result

    def base_oof(self, train, seed):
        oof = np.full((self.ds.n, 3), np.nan)
        for fit, valid, kind in self.splits(train):
            assert not set(fit) & set(valid)
            if kind == 'point':
                assert not set(self.ds.points[fit]) & set(self.ds.points[valid])
            oof[valid] = self.base(fit, seed)[valid]
            self.split_log.append(dict(target_radius=self.radius, seed=seed, kind=kind,
                                       train_indices=list(map(int, fit)), valid_indices=list(map(int, valid))))
        assert np.isfinite(oof[train]).all()
        return oof

    def fitted(self, train, seed):
        key = (tuple(train), seed)
        if key not in self.stack_cache:
            base = self.base(train, seed)
            oof = self.base_oof(train, seed)
            x = np.column_stack([self.ds.xw, base[:, 2]])
            # At meta-fit time every CNN training score is out of fold.
            x[train, -1] = oof[train, 2]
            meta = logistic(x, self.y, train)
            self.stack_cache[key] = np.column_stack([base, meta])
        return self.stack_cache[key]

    def calibration(self, train, seed):
        oof = np.full((self.ds.n, 4), np.nan)
        for fit, valid, kind in self.splits(train):
            assert kind == 'point'  # all outer training sets have >=2 points
            # Whole stack (including its meta learner) excludes validation point.
            oof[valid] = self.fitted(fit, seed)[valid]
        assert np.isfinite(oof[train]).all()
        return oof


def legacy_masks(ds):
    av = pd.read_csv(PIPE / 'audiovisual/selected_candidates.csv')
    wrist = pd.read_csv(PIPE / 'pose/selected_candidates.csv')
    return dict(audio_v1=np.ones(ds.n, dtype=bool),
                audio_v2=ds.audio.kept.to_numpy(dtype=bool),
                audio_motion=ds.c.candidate_id.isin(av.loc[av.method == 'audio_v1_A', 'candidate_id']).to_numpy(),
                audio_wrist=ds.c.candidate_id.isin(wrist.loc[wrist.method == 'audio_v1_wrist', 'candidate_id']).to_numpy())


def refit_rules(ds, train_points):
    """Same hand rules, selected on outer train using the new 95% criterion."""
    train = ds.idx(train_points)
    rules, masks = [], []
    for width, rise, drop, concentration in itertools.product(
            [None, 6, 10, 15, 25, 40, 80], [None, 0, .5, 1, 1.5, 2, 3],
            [None, 0, .5, 1, 2, 3], [None, .3, .5, .7]):
        vals = (width, rise, drop, concentration)
        if sum(v is not None for v in vals) > 3:
            continue
        mask = np.ones(ds.n, dtype=bool)
        for value, feature, sign in zip(vals, ['half_width_ms', 'paired_rise_ln', 'post50_drop_ln',
                                               'energy_concentration_20_100'], [-1, 1, 1, 1]):
            if value is not None:
                mask &= sign * (ds.audio[feature].to_numpy() - value) >= 0
        rules.append(dict(width=width, rise=rise, drop=drop, concentration=concentration))
        masks.append(mask)

    def choose_masks(candidate_masks, configs):
        trials = [ds.metrics(mask, train_points, False) for mask in candidate_masks]
        feasible = [i for i, r in enumerate(trials) if r['recall_clear_f10'] >= .95 - 1e-12]
        chosen = min(feasible, key=lambda i: (trials[i]['n_candidates'], i)) if feasible else min(
            range(len(trials)), key=lambda i: (-trials[i]['recall_clear_f10'], trials[i]['n_candidates'], i))
        return candidate_masks[chosen], dict(**trials[chosen], feasible=bool(feasible), config=configs[chosen])

    v2mask, v2config = choose_masks(masks, rules)
    shape_margin = np.full(ds.n, np.inf)
    for key, feature, sign, scale in [('width', 'half_width_ms', -1, 10), ('rise', 'paired_rise_ln', 1, 1),
                                      ('drop', 'post50_drop_ln', 1, 1), ('concentration', 'energy_concentration_20_100', 1, .2)]:
        value = v2config['config'][key]
        if value is not None:
            shape_margin = np.minimum(shape_margin, sign * (ds.audio[feature].to_numpy() - value) / scale)
    shape = special.expit(shape_margin)
    audio = np.log1p(ds.c.strength.to_numpy().clip(0))
    audio /= max(float(np.median(audio[train])), 1e-8)
    result, configs = dict(audio_v1=np.ones(ds.n, dtype=bool), audio_v2=v2mask), dict(audio_v2=v2config)
    for method, values, shape_weights in [('audio_motion', ds.motion.motion_peak_height.to_numpy(), [0, .5, 1]),
                                         ('audio_wrist', ds.wrist.wrist_peak_height.to_numpy(), [0])]:
        motion = np.log1p(values.clip(0))
        scale = np.nanmedian(motion[train])
        motion = np.nan_to_num(motion / (scale if scale > 0 else 1), nan=0.)
        masks, cfgs = [], []
        for aw, sw, threshold in itertools.product([.25, .5, 1], shape_weights, np.arange(.25, 4.01, .25)):
            masks.append(motion + aw * audio + sw * shape > threshold)
            cfgs.append(dict(audio_weight=aw, shape_weight=sw, threshold=threshold))
        result[method], configs[method] = choose_masks(masks, cfgs)
    return result, configs


def summarize(rows):
    data = pd.DataFrame(rows)
    summary = []
    for keys, group in data.groupby(['evaluation', 'target_radius', 'method'], sort=False):
        row = dict(zip(['evaluation', 'target_radius', 'method'], keys))
        row['n_seeds'] = len(group)
        for metric in METRICS:
            row[metric + '_mean'] = float(group[metric].mean())
            row[metric + '_min'] = float(group[metric].min())
            row[metric + '_max'] = float(group[metric].max())
        summary.append(row)
    return pd.DataFrame(summary)


def table_md(summary, evaluation, radius=10):
    sub = summary[(summary.evaluation == evaluation) & (summary.target_radius == radius)]
    text = '| 方法 | clear ±10F | 候補/秒 | 確認範囲（窓合計） | F1 全±10F |\n|---|---:|---:|---:|---:|\n'
    for r in sub.to_dict('records'):
        def fmt(metric, pct=False):
            factor, digits = (100, 1) if pct else (1, 3)
            val = f"{r[metric+'_mean']*factor:.{digits}f}"
            if r['n_seeds'] > 1:
                val += f" [{r[metric+'_min']*factor:.{digits}f}–{r[metric+'_max']*factor:.{digits}f}]"
            return val + ('%' if pct else '')
        text += f"| {r['method']} | {fmt('recall_clear_f10', True)} | {fmt('candidates_per_sec')} | {fmt('review_window_sum_ratio', True)} | {fmt('f1_all_f10', True)} |\n"
    return text


def verify_outputs(ds, results, predictions, split_log):
    """Recompute saved predictions with the unchanged earlier evaluator."""
    source = PIPE / 'audiovisual/combine_audio_video.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
             node.name in ['match_labels', 'divide', 'evaluate']]
    assert len(nodes) == 3
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    aliases = dict(recall_clear_f6='recall_clear_f6', recall_clear_f10='recall_clear_f10',
                   recall_all_f10='recall_all_f10', precision_proxy_f10='precision_near_label_f10',
                   precision_1to1_f10='precision_one_to_one_f10', candidates_per_sec='candidates_per_sec',
                   review_window_sum_ratio='review_window_sum_ratio', review_window_union_ratio='review_union_ratio')
    checked = 0
    for keys, group in predictions.groupby(['evaluation', 'target_radius', 'method', 'seed']):
        evaluation, radius, method, seed = keys
        points = [9, 10, 11] if evaluation == 'A' else list(range(7, 12))
        assert len(group) == len(ds.idx(points)) and group.candidate_id.is_unique
        assert np.isfinite(group.probability).all()
        assert ((group.probability >= group.threshold) == group.selected).all()
        ids = group.loc[group.selected, 'candidate_id']
        selected = ds.c[ds.c.candidate_id.isin(ids)].copy()
        selected['visual_side'] = ''
        previous, _ = namespace['evaluate'](ds.ints[ds.ints.point.isin(points)],
                                            ds.labels[ds.labels.point.isin(points)], selected)
        saved = results[(results.evaluation == evaluation) & (results.target_radius == radius) &
                        (results.method == method) & (results.seed == seed)].iloc[0]
        for new, old in aliases.items():
            assert np.isclose(saved[new], previous[old], atol=1e-12), (keys, new)
        expected_f1 = 2 * previous['n_matched_all_f10'] / (len(selected) + previous['n_labels_all'])
        assert np.isclose(saved.f1_all_f10, expected_f1, atol=1e-12)
        checked += 1
    for split in split_log:
        fit, valid = split['train_indices'], split['valid_indices']
        assert not set(fit) & set(valid)
        if split['kind'] == 'point':
            assert not set(ds.points[fit]) & set(ds.points[valid])
        else:
            for i in fit:
                nearby = ds.c.iloc[valid]
                nearby = nearby[nearby.rally == ds.c.iloc[i].rally]
                assert not (abs(nearby.candidate_time_sec - ds.c.iloc[i].candidate_time_sec) <= .4).any()
    report = dict(saved_prediction_groups_recomputed=checked, earlier_evaluator_agreement=True,
                  split_nonoverlap_and_purge_passed=True, n_split_checks=len(split_log),
                  previous_evaluator_sha256=sha(source), n_wav_files=len(list(OUT.rglob('*.wav'))))
    assert report['n_wav_files'] == 0
    write_json('verification.json', report)
    return report


def result_judgement(summary):
    text = '## 数値からの判定\n\n'
    for evaluation in ['A', 'B']:
        s = summary[(summary.evaluation == evaluation) & (summary.target_radius == 10)]
        learned = s[s.method.isin(METHODS)]
        best = learned.sort_values('f1_all_f10_mean', ascending=False).iloc[0]
        baseline_name = 'audio_motion' if evaluation == 'A' else 'audio_wrist_refit'
        baseline = s[s.method == baseline_name].iloc[0]
        text += (f"{evaluation}: 学習方式で平均F1最大は {best.method}、"
                 f"F1 {best.f1_all_f10_mean:.1%}、clear再現率 {best.recall_clear_f10_mean:.1%}、"
                 f"候補 {best.candidates_per_sec_mean:.3f}/秒、窓合計 {best.review_window_sum_ratio_mean:.1%}。"
                 f"従来 {baseline_name} はF1 {baseline.f1_all_f10_mean:.1%}、"
                 f"clear {baseline.recall_clear_f10_mean:.1%}、候補 {baseline.candidates_per_sec_mean:.3f}/秒、"
                 f"窓合計 {baseline.review_window_sum_ratio_mean:.1%}。\n\n")
    s = summary[(summary.evaluation == 'B') & (summary.target_radius == 10)]
    base, stack = (s[s.method == m].iloc[0] for m in ['audio_v1', 'stack'])
    text += (f"Bのstackはv1から候補密度を {1-stack.candidates_per_sec_mean/base.candidates_per_sec_mean:.1%}、"
             f"窓合計を {1-stack.review_window_sum_ratio_mean/base.review_window_sum_ratio_mean:.1%} 削減したが、"
             f"clear再現率は {base.recall_clear_f10_mean:.1%} → {stack.recall_clear_f10_mean:.1%}。"
             'Bの4学習方式はいずれも平均clear再現率95%未満で、CNN/stackの最良シードでも95%未満。'
             '**再現率95%を保つという条件付きでは、学習方式が手作りの規則より良くなったとは判定できない。** '
             'AでもCNNの平均F1・候補密度は音+動きを上回らず、特定の好成績シードだけを選ぶべきではない。\n\n')
    sensitivity = summary[(summary.evaluation == 'B') & (summary.target_radius == 6) &
                          (summary.method == 'lr_audio_wrist')].iloc[0]
    wrist = s[s.method == 'audio_wrist_refit'].iloc[0]
    text += (f"参考の±6F教師では音+手首LRがclear {sensitivity.recall_clear_f10_mean:.1%}、"
             f"候補 {sensitivity.candidates_per_sec_mean:.3f}/秒、F1 {sensitivity.f1_all_f10_mean:.1%}。"
             f"しかし従来音+手首の再選択は同じclear {wrist.recall_clear_f10_mean:.1%}で"
             f"候補 {wrist.candidates_per_sec_mean:.3f}/秒、F1 {wrist.f1_all_f10_mean:.1%}であり、"
             'この感度分析からも学習方式の優位は示されない。\n\n')
    return text


def availability_diagnostic(ds):
    rows = []
    for lab in ds.labels.itertuples():
        candidates = ds.c[(ds.c.point == lab.point) & (ds.c.rally == lab.rally)]
        distances = abs(candidates.candidate_frame - lab.contact_frame)
        rows.append(dict(label_id=lab.label_id, point=lab.point, rally=lab.rally,
                         contact_frame=lab.contact_frame, confidence=lab.confidence,
                         nearest_candidate_frames=int(distances.min()),
                         n_candidates_f6=int((distances <= 6).sum()),
                         n_candidates_f10=int((distances <= 10).sum())))
    pd.DataFrame(rows).to_csv(OUT / 'label_candidate_availability.csv', index=False)


def update_ignore():
    ignored = ['# Regenerable cache and generated files larger than 256 KiB', 'logmel_cache.npy']
    ignored += sorted(p.name for p in OUT.iterdir() if p.is_file() and p.stat().st_size > 256*1024 and p.name != 'logmel_cache.npy')
    (OUT / '.gitignore').write_text('\n'.join(ignored) + '\n', encoding='utf-8')


def plots(ds, summary, curves, images):
    colors = dict(zip(METHODS, ['#275d9e', '#d17b12', '#297950', '#9b418a']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, evaluation in zip(axes, ['A', 'B']):
        for method in METHODS:
            g = curves[(curves.evaluation == evaluation) & (curves.method == method)]
            for seed, line in g.groupby('seed'):
                ax.plot(line.candidates_per_sec, line.recall_clear_f10, color=colors[method], alpha=.25, lw=1)
            row = summary[(summary.evaluation == evaluation) & (summary.target_radius == 10) & (summary.method == method)].iloc[0]
            ax.scatter(row.candidates_per_sec_mean, row.recall_clear_f10_mean,
                       color=colors[method], label=method, s=40)
        ax.axhline(.95, color='gray', linestyle='--', lw=1)
        ax.set(xlabel='Candidates / second', ylabel='Clear recall, +/-10 frames',
               title=f'{evaluation}: evaluation-only threshold sweeps', ylim=(0, 1.04))
        ax.legend(fontsize=8)
    fig.savefig(OUT / 'recall_candidate_curves.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, evaluation, metric in zip(axes.flat, ['A', 'A', 'B', 'B'],
                                       ['f1_all_f10', 'review_window_sum_ratio'] * 2):
        sub = summary[(summary.evaluation == evaluation) & (summary.target_radius == 10)]
        x = np.arange(len(sub))
        means, low, high = (sub[metric + suffix].to_numpy() for suffix in ['_mean', '_min', '_max'])
        ax.bar(x, means, color=[colors.get(m, '#999999') for m in sub.method], alpha=.8)
        ax.errorbar(x, means, yerr=np.maximum(0, np.vstack([means-low, high-means])), fmt='none', color='black', capsize=3)
        ax.set_xticks(x, sub.method, rotation=40, ha='right', fontsize=8)
        ax.set(title=f'{evaluation}: {metric}', ylabel='Ratio')
    fig.savefig(OUT / 'comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), constrained_layout=True)
    for row, target in enumerate([1, 0]):
        idx = np.flatnonzero((ds.nearest <= 10) == target)[:4]
        for ax, i in zip(axes[row], idx):
            ax.imshow(images[i, 0], origin='lower', aspect='auto', extent=[-50, 150, 0, 64], cmap='magma')
            ax.axvline(0, color='cyan', lw=.8)
            ax.set_title(f'{ds.c.iloc[i].candidate_id}: target={target}', fontsize=9)
            ax.set(xlabel='Milliseconds from candidate', ylabel='Mel band')
    fig.savefig(OUT / 'logmel_examples.png', dpi=130)
    plt.close(fig)


def readme(summary, counts, thresholds, metadata):
    text = '''# match2 第7〜11点：学習による音候補の選別

## 実行方法

```powershell
& study\\.venv\\Scripts\\python.exe -B -X utf8 outputs\\shot_pipeline_20261117\\audio_learned\\train_hit_classifier.py
```

新しいパッケージは不要。NumPy/SciPy/pandas/matplotlib/PyTorchのみ。既定はCUDA、なければCPU。`--device cpu`、`--epochs 60`、`--ffmpeg C:\\ffmpeg\\bin\\ffmpeg.exe` が使用可能。保存済み結果から説明と診断を更新する場合は `--report-only`（再学習せず入力ハッシュと前回評価関数との一致を再検証）。全出力はこのフォルダ内だけに保存し、元ファイルのハッシュが実行前後で一致することをassertする。WAVはtempfileで復号し削除。npyキャッシュは入力・動画のサイズ/更新時刻・抽出仕様が一致した場合だけ再使用する。git commitしない。

## データと正解の意味

母集団はv1全190候補、6区間（第7点A/B、第8〜11点B）。時刻を動かさず、同一の点/rallyのclearまたはinferredが±10整数F以内なら候補教師1、なければ0。複数候補が同じラベルに対して1になり得る。第8点を含む。±6Fで教師を作り直した感度分析も4モデル・5シードで実施した。ラベルはClaudeが6Fおきの映像から付けたもので独立な打球音正解ではない。「本物の打球」の識別を検証したと断定できない。

| 点 | 候補 | 正例±10F | 負例±10F | 正例±6F | 負例±6F | clear | 全ラベル |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in counts.to_dict('records'):
        text += f"| {r['point']} | {r['n_candidates']} | {r['positive_f10']} | {r['negative_f10']} | {r['positive_f6']} | {r['negative_f6']} | {r['n_clear']} | {r['n_labels']} |\n"
    totals = counts.drop(columns='point').sum()
    text += '| 合計 | ' + ' | '.join(str(int(totals[k])) for k in
                ['n_candidates', 'positive_f10', 'negative_f10', 'positive_f6', 'negative_f6', 'n_clear', 'n_labels']) + ' |\n'
    text += f'''
解析区間の合計は{metadata['duration_sec']:.6f}秒。第7・8点は候補114、clear40、全ラベル59。第9〜11点は候補76、clear18、全ラベル32。評価には追加生成例を入れない。

## モデルと分割

1. `lr_audio`: 音の元特徴12列（strength、半値幅、censored、低/高/paired立上り、低域ピークの2種の遅れ、50ms後の比と減衰、集中度、flatness）。v2のrule_margin/kept、点番号、時刻、正解との距離は入力にしない。
2. `lr_audio_wrist`: 1に手首の最大/第2ピーク、高さ差、側の差、ピークあり人数/側数の6列を加える。選手ID、追跡問題フラグは入力にしない。
3. `cnn`: 候補−50〜+150ms、64メル帯（100–16000Hz）、Hann1024サンプル、2ms刻み、対数パワー64×101。畳込み8/16/24チャネル＋全結合。パラメータ数は後述。正例のみ−5/−2/+2/+5msの実波形シフトと±1dBを学習時に無作為適用。水増しで独立例数は増えない。追加負例は使用しない。
4. `stack`: CNNの確率と2の元特徴をL2ロジスティック回帰に入力。メタ学習用のCNN確率は必ず内部の点単位OOF予測。評価点をCNN・メタ回帰・標準化・しきい値決定のいずれにも使わない。

LRはSciPy L-BFGS-Bで二値交差エントロピー平均＋0.1/2×重み二乗和（切片を除く）を最小化。欠測の中央値、欠測指示列、平均/SDは各学習集合だけで計算。CNNの帯域別平均/SDも学習集合だけで計算。AdamW、lr=.003、weight_decay=.01、dropout=.2、全候補バッチ、固定60epoch（`--epochs`で変更可）、早期停止・評価結果を使った探索なし。5シード17/29/43/71/101でCNNとstackを再学習。LRは決定的なので1行のみ。

**A**: 第7・8点で学習としきい値決め、第9〜11点で評価。**B（主結果）**: 点を丸ごと抜く5分割、各分割の保持/棄却を連結して5点全体の分子/分母で評価する（点別率の単純平均ではない）。

しきい値は外側学習集合の内部OOF予測を使用し、clear±10F再現率95%以上の中で保持候補数最少、同率なら最大しきい値。LR/CNNに加えstack全体も点を抜いてOOF化する。stackのメタ学習ではさらに内部OOFのCNNが必要。Aの最深部だけ1点しか残らないため、その1点を時間順3ブロックに分け、同一rallyで検証候補の±0.4秒の学習候補を除外する。外側評価は常に点単位。水増しは分割後の学習データだけに適用。

95%を満たすしきい値がなければ、学習側clear再現率最大→候補数最少→最大しきい値として未達を明記。評価点で再調整しない。内部OOFは少数点での学習になるため、しきい値が最終モデルと同じ確率分布に移る保証もない。95%条件は評価側95%を保証するものではない。

## 指標

前回と同じ、同一の点/rally内で時間差の小さい順に全ラベルと候補を1対1貪欲照合。±6F/±10Fは別照合し、そのうちclearを数える。同率はラベルF→候補F→label_id→candidate_id。F1=2×全ラベル対応数/(候補数＋全ラベル数)、候補教師の分類F1とは別。適合率目安は±10Fに何らかのラベルがある候補/全候補（多対1可）、1対1適合率も別列。候補/秒は総候補/総区間秒。確認範囲の主指標は候補±10F（連続時間で幅20F）を区間端で切った窓の合計/区間長、重複を数えるため100%超もある。和集合比率もCSVに保存。

貪欲照合は最大対応数を保証せず、候補を落とすとinferredとの競合が変わりclear対応が増える場合もある。したがってしきい値は全ユニーク得点を走査する。

## A：固定分割の評価結果

表は平均 [5シードの最小–最大]。LRと前回方式は単一値。学習教師は±10F。

'''
    text += table_md(summary, 'A')
    text += '''
前回4方式は保存済みの選択候補をそのまま再採点（v1/v2/音+動き/音+手首、交互制約なし）。元の方式と今回ではしきい値の選択目的が異なる。

## B：点ごと1点抜き交差検証（主結果）

'''
    text += table_md(summary, 'B')
    text += '''
Bの `*_refit` は、従来の手作りの規則の探索格子を各外側学習4点だけで選び直し、95%の新しい条件にそろえた比較。音v2の閾値格子（最大3ゲート）、音+動きの音重み/shape重み/閾値格子、音+手首の音重み/閾値格子を使用。尺度は学習点の中央値。音+動きが使うshape規則も学習点で再選択。条件未達なら再現率最大→候補数最少→設定順。v1の候補発見器そのものは元々第7・8点で選ばれた固定母集団であり、再学習していない。このBは固定候補の選別器に対する条件付き評価で、全検出器の完全な未接触交差検証ではない。

### 前回の保存済み規則を5点にそのまま適用した参考値

第7・8点を設定選択に使っているので、この表は交差検証として扱わない。

'''
    text += table_md(summary, 'B_frozen_reference')
    text += '\n' + result_judgement(summary)
    text += '\n## ±6F教師による感度分析（評価幅は同じ±10F）\n\n'
    text += 'A:\n\n' + table_md(summary, 'A', 6) + '\nB:\n\n' + table_md(summary, 'B', 6)
    text += '\n## しきい値の95%条件\n\n'
    for evaluation in ['A', 'B']:
        t = thresholds[(thresholds.evaluation == evaluation) & (thresholds.target_radius == 10)]
        for method, g in t.groupby('method', sort=False):
            text += f"- {evaluation} {method}: 学習OOF条件達成 {int(g.feasible.sum())}/{len(g)}（CNN/stackは分割×シード）。\n"
    text += f'''
全v1候補のclear±10F対応は56/58だが、clear全58件の±10F以内に少なくとも1候補がある。56/58はinferredとの1対1貪欲照合で競合した結果であり、候補の存在だけから決まる絶対上限ではない。例えば第7点を抜いた学習側で全候補のclear再現率が95%未満でも、任意の別の部分集合が95%を達成不能だとは言えない。今回の各モデルのしきい値全走査で条件を満たしたかを `thresholds.csv` のfeasibleで判定し、候補の近傍存在とは区別する。各ラベルの近傍候補数は `label_candidate_availability.csv`。

## 結果の判定と限界

数値に基づく比較は `summary.csv` と `comparison_to_baselines.csv`。比較CSVの差は学習方式−従来方式で、候補/秒・確認範囲は負が削減、再現率・F1は正が改善。両方を同時に見る。再現率を落とした候補削減だけを改善と解釈しない。図の評価側のしきい値走査は診断用で、結果を見て設定を選び直していない。Bの曲線は全OOF確率に共通の仮想しきい値を当てる図で、分割別に学習決定した運用点とは別。

1. 教師は音の真偽を独立に確認したものではない。6〜12Fの時刻誤差、inferred、未ラベル打球があり、±10Fは約167msと広い。近いだけのバウンド・足音・声も正例に入り得る。小さい画像の時間シフトはこの教師誤差を修正しない。
2. 1試合5点、190候補、全91/clear58ラベルだけ。Aの評価はclear18、全32と少なく、1件でclear再現率が5.6ポイント動く。CNNの過学習、点ごとの音響差、極めて小さい内部学習集合による不安定さが大きい。シードの範囲は信頼区間ではなく、独立試合への一般化の誤差でもない。
3. v1の候補発見器と既存の特徴定義には第7・8点での調整、v2の仮説には第9点の観察がある。Bでもこの既存設計由来の楽観性は除けない。すべての区間もラベルで与えられているため、試合全編の偽陽性や区間発見性能は評価していない。
4. 第8点の手前追跡誤りは音の教師には影響しないため含めたが、手首入力は誤りの影響を受ける。欠測指示だけで誤追跡を解消できない。
5. 確認窓の削減は人の確認時間削減の実測ではない。条件をそろえた未使用試合と独立打球ラベルでの評価、同じラベル品質を保つ確認・修正時間の比較が次に必要。本試行だけでは高い有効性や新規性を主張できない。

Decorteほか, CVPRW 2024, *Multi-Modal Hit Detection and Positional Analysis in Padel Competitions* は音に基づく学習モデルで平均F1 92%を報告している。競技、データ量、入力・候補抽出、時刻許容幅、照合条件が異なるため今回のF1と直接比較できない。本実験はそのモデルの再現ではない。一次資料: https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.html

## 出力と実行記録

- `candidate_targets.csv`, `dataset_counts.csv`: 全候補の教師と分母。
- `label_candidate_availability.csv`: 各ラベルの近傍候補の存在と1対1照合の競合を区別する診断。
- `predictions.csv`: 外側評価の全候補、確率、学習側しきい値、保持/棄却（主教師と±6F感度分析）。
- `evaluation.csv`, `summary.csv`, `per_point_evaluation.csv`: シード別、平均/最小/最大、点別の全指標。
- `thresholds.csv`, `rule_refits.json`: 学習側のしきい値・達成可否と規則設定。
- `label_matches.csv`: 各ラベルの1対1対応（±6F/±10F）、不一致も保存。
- `threshold_curves.csv`, `recall_candidate_curves.png`: 評価側で動かした診断用曲線。
- `comparison_to_baselines.csv`, `comparison.png`, `logmel_examples.png`: 比較差分と図。
- `fit_log.json`, `split_audit.json`, `run_metadata.json`: 実際のCNN学習、入れ子分割・非重複assert、入力ハッシュ、実行環境。
- `verification.json`: 保存予測を変更していない前回の評価関数で再採点した一致、点分割の非重複、最深部のパージ幅を検証。
- `logmel_cache.npy/json`: シフト画像も含む特徴キャッシュ。キャッシュ・大きい生成物はこのフォルダの.gitignoreに指定。

CNNパラメータ数: {metadata['cnn_parameters']}。実行デバイス: {metadata['device']}。CNN学習回数: {metadata['cnn_fits']}。epoch: {metadata['epochs']}。実行秒: {metadata['elapsed_sec']:.2f}。入力変更なし: {metadata['inputs_unchanged']}。WAVを保持していない: True。
'''
    (OUT / 'README.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffmpeg', type=Path, default=Path(r'C:\ffmpeg\bin\ffmpeg.exe'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        ds = Dataset()
        metadata = json.loads((OUT / 'run_metadata.json').read_text(encoding='utf-8'))
        assert metadata['input_sha256'] == {str(p.relative_to(PIPE)): sha(p) for p in input_files()}
        results = pd.read_csv(OUT / 'evaluation.csv')
        audit = json.loads((OUT / 'split_audit.json').read_text(encoding='utf-8'))
        verify_outputs(ds, results, pd.read_csv(OUT / 'predictions.csv'), audit)
        availability_diagnostic(ds)
        metadata.setdefault('training_script_sha256', metadata['script_sha256'])
        metadata['script_sha256'] = sha(__file__)
        metadata['post_training_changes'] = 'Report text and label-candidate availability diagnostics only; no model, threshold or evaluation changes.'
        write_json('run_metadata.json', metadata)
        readme(pd.read_csv(OUT / 'summary.csv'), pd.read_csv(OUT / 'dataset_counts.csv'),
               pd.read_csv(OUT / 'thresholds.csv'), metadata)
        update_ignore()
        print('Report regenerated; saved predictions agree with earlier evaluator. No training repeated.')
        return
    t0 = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    before = {str(p.relative_to(PIPE)): sha(p) for p in input_files()}
    ds = Dataset()
    availability_diagnostic(ds)
    images, cache_meta, cache_used = spectrogram_cache(ds, args.ffmpeg)
    targets = ds.c[['candidate_id', 'point', 'rally', 'candidate_frame', 'candidate_time_sec']].copy()
    targets['nearest_label_frames'] = ds.nearest
    targets['target_f10'], targets['target_f6'] = ds.nearest <= 10, ds.nearest <= 6
    targets.to_csv(OUT / 'candidate_targets.csv', index=False)
    counts = []
    for point in range(7, 12):
        ix = ds.idx([point])
        counts.append(dict(point=point, n_candidates=len(ix), positive_f10=int((ds.nearest[ix] <= 10).sum()),
                           negative_f10=int((ds.nearest[ix] > 10).sum()), positive_f6=int((ds.nearest[ix] <= 6).sum()),
                           negative_f6=int((ds.nearest[ix] > 6).sum()), n_clear=int(((ds.lp == point) & ds.clear).sum()),
                           n_labels=int((ds.lp == point).sum())))
    counts = pd.DataFrame(counts)
    counts.to_csv(OUT / 'dataset_counts.csv', index=False)
    rows, point_rows, prediction_rows, threshold_rows, match_rows, curve_rows, fit_log, split_log = ([] for _ in range(8))
    refits = []
    frozen = legacy_masks(ds)
    splits = [('A', '07_08_to_09_11', [7, 8], [9, 10, 11])] + [
        ('B', f'leave_{p:02}', [q for q in range(7, 12) if q != p], [p]) for p in range(7, 12)]

    def record(evaluation, radius, method, seed, selected, points):
        common = dict(evaluation=evaluation, target_radius=radius, method=method, seed=seed)
        rows.append(dict(**common, **ds.metrics(selected, points)))
        for p in points:
            point_rows.append(dict(**common, point=p, **ds.metrics(selected, [p])))
        for tol in (6, 10):
            matches = {li: (ci, delta) for li, ci, delta in ds.match(selected, points, tol)}
            for li in np.flatnonzero(np.isin(ds.lp, points)):
                ci, delta = matches.get(li, (None, None))
                lab = ds.labels.iloc[li]
                match_rows.append(dict(**common, label_id=lab.label_id, point=int(lab.point), rally=lab.rally,
                                       confidence=lab.confidence, contact_frame=int(lab.contact_frame), tolerance=tol,
                                       candidate_id=ds.c.iloc[ci].candidate_id if ci is not None else '', delta_frames=delta))

    # Legacy A is exactly the earlier frozen methods. B refits only selectors.
    baseline_b = {method: np.zeros(ds.n, dtype=bool) for method in OLD}
    for evaluation, fold, train_points, test_points in splits:
        masks, configs = (frozen, {}) if evaluation == 'A' else refit_rules(ds, train_points)
        refits.append(dict(evaluation=evaluation, fold=fold, train_points=train_points, test_points=test_points, configs=configs))
        for method in OLD:
            if evaluation == 'A':
                record('A', 10, method, -1, masks[method], test_points)
            else:
                test = ds.idx(test_points)
                baseline_b[method][test] = masks[method][test]
    for method in OLD:
        record('B', 10, method + ('_refit' if method != 'audio_v1' else ''), -1, baseline_b[method], list(range(7, 12)))
        record('B_frozen_reference', 10, method, -1, frozen[method], list(range(7, 12)))

    for radius in (10, 6):
        learner = Learner(ds, images, radius, args.device, args.epochs)
        for seed in SEEDS:
            selected_b = {m: np.zeros(ds.n, dtype=bool) for m in METHODS}
            scores_b = {m: np.zeros(ds.n) for m in METHODS}
            for evaluation, fold, train_points, test_points in splits:
                train, test = ds.idx(train_points), ds.idx(test_points)
                assert not set(train_points) & set(test_points)
                calibration = learner.calibration(train, seed)
                scores = learner.fitted(train, seed)
                print(f'target={radius} seed={seed} {fold} fits={len(learner.fit_log)} elapsed={time.perf_counter()-t0:.1f}s', flush=True)
                for mi, method in enumerate(METHODS):
                    if method.startswith('lr') and seed != SEEDS[0]:
                        continue
                    # Calibration and its complete nested pipeline exclude test.
                    training_scores = np.zeros(ds.n)
                    training_scores[train] = calibration[train, mi]
                    chosen = ds.choose(training_scores, train_points)
                    threshold_rows.append(dict(evaluation=evaluation, target_radius=radius, seed=seed, method=method,
                                               fold=fold, train_points=';'.join(map(str, train_points)),
                                               test_points=';'.join(map(str, test_points)),
                                               all_candidate_train_recall=ds.metrics(np.ones(ds.n, dtype=bool), train_points, False)['recall_clear_f10'], **chosen))
                    selected = scores[:, mi] >= chosen['threshold']
                    for i in test:
                        prediction_rows.append(dict(evaluation=evaluation, target_radius=radius, seed=seed, method=method,
                                                    fold=fold, candidate_id=ds.c.iloc[i].candidate_id, point=int(ds.points[i]),
                                                    probability=float(scores[i, mi]), threshold=chosen['threshold'],
                                                    selected=bool(selected[i]), target=int(learner.y[i])))
                    if evaluation == 'A':
                        record(evaluation, radius, method, seed, selected, test_points)
                    else:
                        selected_b[method][test] = selected[test]
                        scores_b[method][test] = scores[test, mi]
                    if radius == 10 and evaluation == 'A':
                        for threshold in np.linspace(0, 1, 101):
                            curve_rows.append(dict(evaluation=evaluation, seed=seed, method=method, threshold=threshold,
                                                   **ds.metrics(scores[:, mi] >= threshold, test_points)))
            for method in METHODS:
                if method.startswith('lr') and seed != SEEDS[0]:
                    continue
                record('B', radius, method, seed, selected_b[method], list(range(7, 12)))
                if radius == 10:
                    for threshold in np.linspace(0, 1, 101):
                        curve_rows.append(dict(evaluation='B', seed=seed, method=method, threshold=threshold,
                                               **ds.metrics(scores_b[method] >= threshold, list(range(7, 12)))))
        fit_log.extend(learner.fit_log)
        split_log.extend(learner.split_log)
        del learner
    results = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize(rows)
    thresholds = pd.DataFrame(threshold_rows)
    curves = pd.DataFrame(curve_rows)
    for name, table in [('evaluation.csv', results), ('summary.csv', summary), ('thresholds.csv', thresholds),
                        ('predictions.csv', predictions), ('label_matches.csv', pd.DataFrame(match_rows)),
                        ('per_point_evaluation.csv', pd.DataFrame(point_rows)), ('threshold_curves.csv', curves)]:
        table.to_csv(OUT / name, index=False)
    comparison = []
    for evaluation in ['A', 'B']:
        sub = summary[(summary.evaluation == evaluation) & (summary.target_radius == 10)]
        for method in METHODS:
            a = sub[sub.method == method].iloc[0]
            for b in sub[~sub.method.isin(METHODS)].to_dict('records'):
                comparison.append(dict(evaluation=evaluation, learned=method, baseline=b['method'], **{
                    m + '_delta': float(a[m+'_mean'] - b[m+'_mean']) for m in METRICS}))
    pd.DataFrame(comparison).to_csv(OUT / 'comparison_to_baselines.csv', index=False)
    write_json('rule_refits.json', refits)
    write_json('fit_log.json', fit_log)
    write_json('split_audit.json', split_log)
    plots(ds, summary, curves, images)
    verification = verify_outputs(ds, results, predictions, split_log)
    after = {str(p.relative_to(PIPE)): sha(p) for p in input_files()}
    assert before == after
    # Earlier table equivalence is an independent regression check.
    expected = dict(audio_v1=(76, 18, 32), audio_v2=(58, 17, 28), audio_motion=(63, 18, 32))
    for method, values in expected.items():
        r = results[(results.evaluation == 'A') & (results.method == method)].iloc[0]
        assert tuple(int(r[k]) for k in ['n_candidates', 'n_matched_clear_f10', 'n_matched_all_f10']) == values
    metadata = dict(device=args.device, epochs=args.epochs, seeds=list(SEEDS),
                    cnn_parameters=sum(p.numel() for p in TinyCNN().parameters()), cnn_fits=len(fit_log),
                    elapsed_sec=time.perf_counter()-t0, input_sha256=before, inputs_unchanged=before == after,
                    script_sha256=sha(__file__), cache_used=cache_used, cache_sha256=sha(OUT/'logmel_cache.npy'),
                    duration_sec=float(ds.ints.duration_sec.sum()), torch_version=torch.__version__,
                    numpy_version=np.__version__, cuda_available=torch.cuda.is_available(),
                    wav_retained=False, nested_crossfit=True, primary_label_radius=10,
                    reference_label_radius=6, baseline_A_regression_passed=True,
                    hyperparameters_tuned_on_test=False)
    metadata['verification_passed'] = verification['earlier_evaluator_agreement']
    write_json('run_metadata.json', metadata)
    readme(summary, counts, thresholds, metadata)
    update_ignore()
    print(summary[(summary.target_radius == 10)][['evaluation', 'method', 'recall_clear_f10_mean',
          'candidates_per_sec_mean', 'review_window_sum_ratio_mean', 'f1_all_f10_mean']].to_string(index=False), flush=True)
    print(f'COMPLETE: {metadata["elapsed_sec"]:.2f}s, CNN fits={len(fit_log)}', flush=True)


if __name__ == '__main__':
    main()
