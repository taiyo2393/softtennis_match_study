"""Adapt already used local review UI to52 new cases without starting it."""
from pathlib import Path
import json
from integrate_044_reviewed_tracks import ROOT, sha

source = ROOT / 'serve_short_ball_sequence_review.py'
target = ROOT / 'serve_reserved_ball_sequence_review.py'
assert not target.exists()
text = source.read_text(encoding='utf-8')
text = text.replace('Prediction-blind local annotation of24 fixed sequence times', 'Prediction-blind local annotation of52 reserved sequence times')
text = text.replace("OUT=ROOT/'ball_short_sequence_evaluation'", "OUT=ROOT.parent/'ball_quality_20261104'")
text = text.replace('==24', '==52').replace('==26', '==52').replace('annotation_cases=24', 'annotation_cases=52')
text = text.replace('[AB]\\d{2}', '[ABCD]\\d{2}')
assert text != source.read_text(encoding='utf-8') and '[AB]\\d{2}' not in text
compile(text,str(target),'exec')
target.write_text(text,encoding='utf-8')
out=ROOT.parent/'ball_quality_20261104'
state=dict(status='review_server_prepared_not_started',source_template=str(source),source_template_sha256=sha(source),
           server_script=str(target),server_script_sha256=sha(target),
           adaptations=['52pending cases','ABCD case IDs','new output directory only'],real_labels_written=False)
(out/'review_tools_manifest.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
print(json.dumps(state,indent=2))
