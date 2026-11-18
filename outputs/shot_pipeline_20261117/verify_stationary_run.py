"""Independent numeric audit of saved stationary runs, masks and protected files."""
import hashlib,json,math,pickle,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];PIPE=Path(__file__).resolve().parent;OUT=PIPE/'tracking_v2'
def main():
    fps=60000/1001
    report=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    run_count=0
    for point in range(7,12):
        with (OUT/f'point_{point:02d}/candidate_cache.pkl').open('rb') as stream:cache=pickle.load(stream)
        xy={(f['source_frame'],r['track_id']):uv for f in cache['frames'] for r,uv in zip(f['candidates'],f['uv'])}
        rows=pd.read_csv(OUT/f'point_{point:02d}/player_court_proxies.csv')
        for stat in [s for s in report['excluded_candidates'] if s['point']==point]:
            total=0;previous=-1
            for run in stat['stationary_runs']:
                first,last=run['first_frame'],run['last_frame'];assert first>previous;previous=last
                positions=np.asarray([xy.get((f,stat['track_id']),[np.nan,np.nan]) for f in range(first,last+1)])
                good=np.isfinite(positions).all(axis=1)
                center=np.nanmedian(positions[good],axis=0)
                assert center[0]<-.05 or center[0]>1.05
                radius=np.sqrt(np.sum(((positions-center)*[10.97,23.77])**2,axis=1))
                inside=good&(radius<=.7)
                assert inside[0] and inside[-1]
                current=longest=0
                for valid in inside:
                    current=0 if valid else current+1;longest=max(longest,current)
                assert longest<=math.floor(.2*fps)
                assert last-first+1>=math.ceil(2*fps)
                assert run['duration_frames']==last-first+1 and run['inlier_frames']==int(inside.sum())
                assert np.allclose(center,[run['median_u'],run['median_v']],rtol=0,atol=1e-12)
                total+=last-first+1;run_count+=1
            frames=sorted(f for f,t in xy if t==stat['track_id'])
            ratio=total/(frames[-1]-frames[0]+1)
            assert abs(ratio-stat['stationary_fraction'])<1e-12
            assert (stat['exclusion_mode']=='whole_track')==(ratio>=.30)
            expected=([[frames[0],frames[-1]]] if ratio>=.30 else
                [[max(frames[0],r['first_frame']-30),min(frames[-1],r['last_frame']+30)] for r in stat['stationary_runs']])
            assert stat['exclusion_ranges']==expected
            for a,b in stat['exclusion_ranges']:
                assert rows[(rows.track_id==stat['track_id'])&rows.source_frame.between(a,b)].empty
    comparison=pd.read_csv(OUT/'comparison.csv')
    control=comparison[comparison.comparison=='original_vs_unfiltered_replay']
    assert control[['position_difference_ge_0_5m','box_changed_frames','removed_frames','added_frames']].eq(0).all().all()
    protected=json.loads((OUT/'protected_all_before_stationary_run.json').read_text(encoding='utf-8'))
    for relative,digest in protected.items():
        with (ROOT/relative).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,relative
    actual={p.relative_to(ROOT).as_posix() for d in [PIPE/'tracking',PIPE/'pose',ROOT/'outputs/ball_quality_20261111'] for p in d.rglob('*') if p.is_file()}
    assert actual=={Path(p).as_posix() for p in protected},'Protected file set changed'
    large=[p for d in [OUT,PIPE/'court_view_p07_11'] for p in d.rglob('*') if p.is_file() and p.stat().st_size>256*1024]
    proc=subprocess.run(['git','check-ignore','--stdin','-z'],
        input=('\0'.join(p.relative_to(ROOT).as_posix() for p in large)+'\0').encode(),capture_output=True,cwd=ROOT)
    assert set(proc.stdout.decode().rstrip('\0').split('\0'))=={p.relative_to(ROOT).as_posix() for p in large}
    audit=dict(protected_files_sha256_unchanged=len(protected),protected_file_set_unchanged=True,
        stationary_runs_independently_verified=run_count,excluded_tracks=len(report['excluded_candidates']),
        original_unfiltered_replay_exact=True,exclusion_masks_verified=True,large_files_ignored=len(large))
    (OUT/'stationary_run_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
