"""Run all three analyses with this interpreter and audit pre-existing inputs."""
from pathlib import Path
import subprocess
import sys
from common import HERE, PIPE, ROOT, sha, dump, ignore_large


def snapshot():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in PIPE.rglob('*')
            if p.is_file() and not p.is_relative_to(HERE)}


def main():
    before=snapshot()
    extra=[ROOT/'outputs/match2_point_inventory_20261117/point_inventory.csv',
           ROOT/'outputs/ball_quality_20261110/match2_camera_transfer_pilot/protocol.json']
    before.update({p.relative_to(ROOT).as_posix():sha(p) for p in extra})
    jobs=['shot_table/build_shot_table.py','audio_eval/evaluate_audio_on_human.py',
          'net_motion/classify_net_motion_on_human.py']
    for job in jobs:
        result=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(HERE/job)],
                              cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (HERE/(job.split('/')[0]+'/execution.log')).write_text(result.stdout,encoding='utf-8')
        print(result.stdout,flush=True)
        assert result.returncode==0, f'{job} failed with exit {result.returncode}'
    after=snapshot()
    after.update({p.relative_to(ROOT).as_posix():sha(p) for p in extra})
    assert before==after,'Pre-existing input or output changed'
    dump(HERE/'preservation_manifest.json',dict(python_executable=sys.executable,
        old_files_unchanged=True,old_files_count=len(before),old_files_sha256=before))
    ignored=ignore_large()
    print(f'Completed all analyses; {len(before)} pre-existing files unchanged; {len(ignored)} large files ignored.',flush=True)


if __name__=='__main__':
    main()
