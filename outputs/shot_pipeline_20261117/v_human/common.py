"""Shared read-only inputs and writers confined to v_human."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
ROOT = PIPE.parent.parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def labels():
    import pandas as pd
    d = pd.read_csv(PIPE / 'labels_human/human_labels_20261118.csv', keep_default_na=False)
    d = d[d.condition == 'without_candidates'].copy()
    assert len(d) == 32 and d.status.eq('completed').all()
    assert d.groupby('point').size().to_dict() == {7: 8, 8: 8, 9: 8, 10: 4, 11: 4}
    assert not d.duplicated(['point', 'frame']).any()
    assert d.side.isin(['N', 'F']).all()
    return d.sort_values(['point', 'rally_no', 'frame']).reset_index(drop=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def ignore_large():
    large = sorted(p for p in HERE.rglob('*') if p.is_file() and p.stat().st_size > 256 * 1024)
    (HERE / '.gitignore').write_text('# Generated files over 256 KiB\n' +
        '\n'.join('/' + p.relative_to(HERE).as_posix() for p in large) + '\n', encoding='utf-8')
    return [p.relative_to(HERE).as_posix() for p in large]


def md_table(df):
    # Avoid an optional tabulate dependency.
    rows = [list(df.columns)] + df.fillna('').astype(str).values.tolist()
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |',
                      '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'] +
                     ['| ' + ' | '.join(r) + ' |' for r in rows[1:]])
