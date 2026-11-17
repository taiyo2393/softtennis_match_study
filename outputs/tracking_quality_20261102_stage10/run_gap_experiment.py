import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import gap_pair_tracking as g

ROOT=Path(__file__).resolve().parent


def source(tag):
    p=ROOT.parent
    if tag=='003':return p/'tracking_quality_20261102_stage9/recompute/tracking_quality_20261102_stage4/players_003_temporal'
    if tag in ('007','025'):return p/'tracking_quality_20261102_stage8/tracking_quality_20261102_stage4'/f'players_{tag}_temporal'
    return p/'tracking_quality_20261102_stage4'/f'players_{tag}_temporal'


def main():
    p=argparse.ArgumentParser();p.add_argument('--tags',nargs='+',default=['025']);tags=p.parse_args().tags
    out=ROOT/'experiment/tracking_quality_20261102_stage7';out.mkdir(parents=True,exist_ok=True)
    plan=dict(parameters=g.P,code_sha256=hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest(),
        all_planned_tags=['025','002','003','005','007','048'],
        scope='Development short-gap linking; preserve prior reviewed outputs and compare regressions')
    planpath=ROOT/'experiment_protocol.json'
    if planpath.exists():assert json.loads(planpath.read_text(encoding='utf-8'))==plan
    else:planpath.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    g.OUT=out
    for tag in tags:
        original=source(tag)
        dest=ROOT/'experiment/tracking_quality_20261102_stage4'/f'players_{tag}_temporal'
        dest.mkdir(parents=True,exist_ok=True)
        for filename in ('summary.json','events.json','selected_four.csv'):
            target=dest/filename
            if target.exists():assert target.read_bytes()==(original/filename).read_bytes()
            else:shutil.copyfile(original/filename,target)
        result=out/f'players_{tag}_global'
        if not (result/'summary.json').exists():
            sys.argv=[g.__file__,'--tag',tag];g.main()
        print('Completed',tag,flush=True)


if __name__=='__main__':main()
