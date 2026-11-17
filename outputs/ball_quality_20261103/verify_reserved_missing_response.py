"""Check saved raw responses independently of model inference and derived CSVs."""
import json,math,shutil
from pathlib import Path
import numpy as np
from integrate_044_reviewed_tracks import ROOT,read,load,sha
from compare_ball_candidate_decoders import components

OUT=ROOT.parent/'ball_quality_20261109/missing_response_diagnosis'
BASE=ROOT.parent/'ball_quality_20261104'
protocol=load(OUT/'protocol.json');result=load(OUT/'result.json')
script=ROOT/'diagnose_reserved_missing_response.py'
assert sha(script)==protocol['script_sha256']
shutil.copy2(script,OUT/'executed_source.py')
with np.load(OUT/'replayed_heatmaps.npz') as data:
    maps={(str(cid),int(t)):hm.astype(np.float32) for cid,t,hm in zip(data['candidate_ids'],data['local_frames'],data['heatmaps'],strict=True)}
assert len(maps)==84
cache=load(BASE/'frozen_component_cache.json')
assert all(components(hm)==cache[cid][str(t)] for (cid,t),hm in maps.items())
labels={g['case_id']:g for g in read(BASE/'completed_review/manual_sequence_labels.csv')}
rows=read(OUT/'visible_response_records.csv');assert len(rows)==45
yy,xx=np.mgrid[:288,:512]
counts={'correct_control':0,'missing_visible':0};outside=0
for r in rows:
    g=labels[r['case_id']];assert g['phase_status']=='play' and g['truth_visibility']=='visible'
    assert all(r[k]==g[k] for k in ['candidate_id','local_frame','source_frame','truth_x_px','truth_y_px'])
    hm=maps[g['candidate_id'],int(g['local_frame'])]
    circle=(xx*3.75-float(g['truth_x_px']))**2+(yy*3.75-float(g['truth_y_px']))**2<=256
    peak=float(hm[circle].max());global_peak=float(hm.max())
    assert float(r['peak_within16px'])==peak and float(r['global_peak'])==global_peak
    assert math.isclose(float(r['local_peak_gap_to03']),.3-peak,abs_tol=1e-7)
    assert int(r['pixels_above03_near16'])==int((hm[circle]>.3).sum())
    assert int(r['full_components_above03'])==len(components(hm))
    counts[r['outcome']]+=1
    if r['outcome']=='missing_visible':
        assert global_peak<=.3 and len(components(hm))==0
        gy,gx=np.unravel_index(np.argmax(hm),hm.shape)
        distance=math.hypot(gx*3.75-float(g['truth_x_px']),gy*3.75-float(g['truth_y_px']))
        assert distance==float(r['global_peak_distance_from_manual_px'])
        outside+=distance>16
assert counts=={'correct_control':33,'missing_visible':12} and outside==4
missing=[r for r in rows if r['outcome']=='missing_visible']
ordered=sorted(missing,key=lambda r:(float(r['peak_within16px']),r['case_id']))
assert result['examples']==[ordered[0]['case_id'],ordered[-1]['case_id']]==['B12','B03']
assert result['missing_peak_range']==[float(ordered[0]['peak_within16px']),float(ordered[-1]['peak_within16px'])]
assert all(sha(Path(p))==h for p,h in result['protected_input_hashes'].items())
images=load(OUT/'input_image_hashes.json');assert all(sha(Path(p))==h for p,h in images.items())
assert sha(OUT/'response_extremes.png')==result['figure_sha256']
verification=dict(status='passed',maps_reproducing_frozen_components=84,human_center_response_records_checked=45,
    missing_maps_without_candidates=12,missing_global_peaks_outside16px=4,input_images_unchanged=len(images),
    original_protocol_and_evaluation_preserved=True,models_and_canonical_outputs_unchanged=True,
    no_threshold_change=True,no_new_labels=True,no_training=True,
    artifacts={name:sha(OUT/name) for name in ['protocol.json','result.json','replayed_heatmaps.npz','visible_response_records.csv',
        'missing_response_records.csv','window_response_summary.csv','response_extremes.png','executed_source.py']},
    script_sha256=sha(Path(__file__)))
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in verification.items() if k!='artifacts'},indent=2))
