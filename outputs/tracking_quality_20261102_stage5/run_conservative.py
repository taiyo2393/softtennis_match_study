import hashlib
import json
from pathlib import Path

import run_guarded as first
from conservative_pair_tracker import ConservativePairTracker,PAIR_POLICY
from identity_guard import GUARD_PARAMETERS,old

HERE = Path(__file__).resolve().parent
OUT = HERE/'conservative'


def main():
    OUT.mkdir(exist_ok=True)
    first.OUT = OUT
    first.runner.OUT = OUT
    first.runner.TemporalTracker = ConservativePairTracker
    first.runner.PARAMETERS = dict(old.PARAMETERS,identity_guard=GUARD_PARAMETERS,pair_policy=PAIR_POLICY)
    baseline = json.loads((first.STAGE4/'experiment_protocol.json').read_text(encoding='utf-8'))
    files = [HERE/'identity_guard.py',HERE/'conservative_pair_tracker.py',HERE/'run_conservative.py',
             HERE/'run_guarded.py',first.STAGE4/'temporal_tracker.py']
    protocol = dict(parameters=first.runner.PARAMETERS,intervals=baseline['intervals'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='Conservative abstention development after human-reported F1/F2 swap. No identity accuracy claim.')
    (OUT/'experiment_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    reports = []
    for interval in baseline['intervals']:
        prior = json.loads((first.STAGE4/f'players_{interval["tag"]}_temporal/summary.json').read_text(encoding='utf-8'))
        assert not (OUT/f'players_{interval["tag"]}_temporal/summary.json').exists(),'Use a fresh experiment folder; completed artifacts are preserved.'
        summary = first.runner.process(interval,Path(prior['input_csv']))
        tracker = ConservativePairTracker.instances[-1]
        first.render_review(summary,tracker)
        summary['pair_policy'] = dict(parameters=PAIR_POLICY,initial_pair_color_similarity=tracker.similar_pairs)
        (OUT/f'players_{interval["tag"]}_temporal/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        reports.append(summary)
        (OUT/'experiment_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    for path,digest in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


if __name__ == '__main__':
    main()
