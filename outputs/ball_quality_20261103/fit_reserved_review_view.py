"""Presentation-only full-court fit; preserve original frozen HTML and server."""
import json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT, sha
OUT=ROOT.parent/'ball_quality_20261104'
source=OUT/'annotation.html';target=OUT/'annotation_fit.html'
assert not target.exists()
text=source.read_text(encoding='utf-8')
text=text.replace("canvas.style.width='100%'", "canvas.style.width='min(100%, calc(70vh * 16 / 9))'")
text=text.replace("canvas.style.width=100*zoom+'%'", "canvas.style.width=zoom===1?'min(100%, calc(70vh * 16 / 9))':'200%'")
text=text.replace('canvas{display:block;width:100%;', 'canvas{display:block;width:min(100%,calc(70vh * 16 / 9));')
assert text!=source.read_text(encoding='utf-8')
target.write_text(text,encoding='utf-8')
server=ROOT/'serve_reserved_ball_sequence_review.py';fit=ROOT/'serve_reserved_ball_sequence_review_fit.py'
assert not fit.exists()
code=server.read_text(encoding='utf-8').replace("OUT/'annotation.html'", "OUT/'annotation_fit.html'")
code=code.replace("m=load(OUT/'manifest.json');cases=", "view=load(OUT/'view_revision.json');assert sha(OUT/'annotation_fit.html')==view['fit_html_sha256'];m=load(OUT/'manifest.json');cases=")
compile(code,str(fit),'exec');fit.write_text(code,encoding='utf-8')
revision=dict(status='presentation_only_fit_revision',original_html_sha256=sha(source),fit_html_sha256=sha(target),
    original_server_sha256=sha(server),fit_server=str(fit),fit_server_sha256=sha(fit),
    change='Initial and normal zoom image fits viewport height70vh and container width, preserving1920:1080.2x mode intentionally scrolls. Fullscreen letterbox coordinate math unchanged.',
    data_or_predictions_changed=False,labels_changed=False)
(OUT/'view_revision.json').write_text(json.dumps(revision,indent=2),encoding='utf-8')
print(json.dumps(revision,indent=2))
