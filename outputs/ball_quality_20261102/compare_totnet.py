"""Development comparison; same-match reused labels are not an external test."""
import argparse
from collections import OrderedDict
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'totnet_comparison'
sys.path.insert(0, str(ROOT / 'third_party/python_deps'))
sys.path.insert(0, str(ROOT.parent / 'tracking_quality_20261102'))
from ball_alignment import SOURCE, read, evaluate

LABELS = SOURCE / 'annotations/balltrack_groundtruth_match1_pilot.csv'
BASELINE = ROOT.parent / 'tracking_quality_20261102/ball_aligned'
THRESHOLDS = [0., .0001, .0003, .001, .003, .01, .03, .1, .3, 1.]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Do not overwrite an existing protocol'
    setup = json.loads((ROOT / 'totnet_setup_report.json').read_text())
    save('protocol.json', dict(
        model='official TOTNet tennis 5-frame ch64', checkpoint_sha256=setup['checkpoint_sha256'],
        source_revision=setup['revision'], label_sha256=sha(LABELS),
        input_color='BGR as Tennis_Dataset training; demo RGB is not used',
        resize='cv2.INTER_LANCZOS4 to width512 height288, as training Resize default',
        normalize=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),
        frame_window='t-4,t-3,t-2,t-1,t; repeat frame0 if negative',
        inference='float32 eval inference_mode; no augmentation; batch4',
        position='global heatmap argmax scaled x*W/512,y*H/288; no interpolation',
        visibility='peak softmax >= threshold; score is not calibrated visibility probability',
        thresholds=THRESHOLDS,
        selection='Maximize validation localized TP subject to nonlocalized outputs <= baseline; ties fewer outputs then higher threshold',
        adoption='Require test localized TP increase and nonlocalized outputs not above baseline; provisional development decision only',
        tolerance_px=16, phases=['validation 001-010','test 011-020'],
        warning='Previously used same-match test; annotation independence/provenance unverified. New external evaluation required.'))
    # The inference plan includes frame identities only, never truth positions or visibility.
    rows=read(LABELS)
    for phase in ('validation','test'):
        save(f'{phase}_frames.json', [dict(candidate_id=r['candidate_id'],local_frame=int(r['local_frame']))
                                      for r in rows if r['evaluation_split']==phase])
    print('Protocol fixed before model comparison', flush=True)


def model():
    path=ROOT/'model_weights/totnet_tennis_best.pth'
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(path)==protocol['checkpoint_sha256']
    assert set(torch.serialization.get_unsafe_globals_in_checkpoint(path))=={'easydict.EasyDict'}
    with torch.serialization.safe_globals([(OrderedDict,'easydict.EasyDict')]):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
    config=ckpt['configs']; attrs=dict(config);attrs.update(vars(config))
    assert attrs['num_frames']==5 and attrs['bidirect'] is False
    spec=importlib.util.spec_from_file_location('official_totnet',ROOT/'third_party/TOTNet/src/model/TOTNet.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    net=mod.TemporalConvNet(input_shape=(288,512),spatial_channels=64,num_frames=5)
    weights={k.removeprefix('module.'):v for k,v in ckpt['state_dict'].items()}
    net.load_state_dict(weights,strict=True)
    assert torch.cuda.is_available()
    return net.cuda().eval()


def input_frames(cid,t):
    ims=[]; dims=None
    for f in range(t-4,t+1):
        p=SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{max(0,f):06d}.jpg'
        im=cv2.imread(str(p));assert im is not None,str(p)
        h,w=im.shape[:2]
        assert dims is None or dims==(w,h)
        dims=(w,h)
        im=cv2.resize(im,(512,288),interpolation=cv2.INTER_LANCZOS4)
        im=(im/255.-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
        ims.append(np.moveaxis(im,-1,0).astype(np.float32))
    return np.stack(ims),dims


def infer(phase):
    target=OUT/f'{phase}_raw.csv'
    assert not target.exists(),'Preserve earlier inference'
    if phase=='test':assert (OUT/'selection.json').exists(),'Freeze validation choice first'
    net=model();plan=json.loads((OUT/f'{phase}_frames.json').read_text())
    rows=[];started=time.perf_counter();checked=False
    with torch.inference_mode():
        for start in range(0,len(plan),4):
            batch=plan[start:start+4]
            inputs=[input_frames(p['candidate_id'],p['local_frame']) for p in batch]
            tensor=torch.from_numpy(np.stack([im for im,_ in inputs])).cuda()
            output=net(tensor)
            assert output.shape==(len(batch),512*288) and torch.isfinite(output).all()
            assert torch.allclose(output.sum(1),torch.ones(len(batch),device='cuda'),atol=1e-4)
            if not checked:
                single=net(tensor[:1])
                assert torch.allclose(single,output[:1],atol=1e-5,rtol=1e-3)
                assert int(single.argmax())==int(output[0].argmax())
                checked=True
            peaks,indices=output.max(1)
            for p,(im,(w,h)),peak,index in zip(batch,inputs,peaks.tolist(),indices.tolist()):
                y,x=divmod(index,512)
                rows.append(dict(**p,x_px=x*w/512,y_px=y*h/288,peak=peak,
                                 width=w,height=h,input_start=max(0,p['local_frame']-4),input_end=p['local_frame']))
            if start%20==0:print(f'{phase}: {min(start+4,len(plan))}/{len(plan)}',flush=True)
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(f'{phase}_inference.json',dict(frames=len(rows),seconds=time.perf_counter()-started,
        batch_single_consistency_passed=checked,protocol_sha256=sha(OUT/'protocol.json'),
        script_sha256=sha(Path(__file__)),raw_sha256=sha(target)))


def predictions(rows,threshold):
    pred={}
    for r in rows:
        origin=float(r['x_px'])==0 and float(r['y_px'])==0
        p=dict(r,visible=int(not origin and float(r['peak'])>=threshold),confidence=r['peak'],threshold=threshold)
        if not p['visible']:p.update(x_px=0,y_px=0)
        pred.setdefault(r['candidate_id'],{})[int(r['local_frame'])]=p
    return pred


def correct_origin():
    """Repair a decoder convention using validation only; preserve original evidence."""
    assert not (OUT/'test_raw.csv').exists()
    archive=OUT/'initial_peak_only_decoder';archive.mkdir(exist_ok=False)
    for name in ('protocol.json','selection.json'):
        shutil.copy2(OUT/name,archive/name)
    protocol=json.loads((OUT/'protocol.json').read_text())
    protocol['visibility']='(x,y)!=(0,0) and peak softmax >= threshold; origin is missing-coordinate sentinel'
    protocol['decoder_correction']=dict(
        reason='Initial adapter counted the model origin sentinel as visible. Detected on validation; corrected before any test inference.',
        evidence='src/losses_metrics/metrics.py TrackNet-style coordinate conventions; src/losses_metrics/losses.py HeatmapBallDetectionLoss2DWeighted clamps absent coordinates to origin.',
        changes='Decoder only. Same raw validation predictions, preprocessing, threshold grid and selection objective.',
        initial_protocol_sha256=sha(archive/'protocol.json'),raw_inference_protocol_retained='validation_inference.json')
    save('protocol.json',protocol)
    (OUT/'selection.json').unlink()  # Original selection preserved in initial_peak_only_decoder.
    print('Origin decoding corrected; initial protocol and selection archived; test remains unseen',flush=True)


def baseline(labels):
    return {cid:{int(r['local_frame']):r for r in read(BASELINE/cid/'balltrack_predictions.csv')}
            for cid in sorted({r['candidate_id'] for r in labels})}


def nonlocalized(metrics):
    c=metrics['counts'];return c.get('pred_visible',0)-c.get('localized_tp_16px',0)


def select():
    assert not (OUT/'selection.json').exists() and not (OUT/'test_raw.csv').exists()
    assert sha(LABELS)==json.loads((OUT/'protocol.json').read_text())['label_sha256']
    labels=[r for r in read(LABELS) if r['evaluation_split']=='validation']
    raw=read(OUT/'validation_raw.csv'); bm=evaluate(labels,baseline(labels),'validation')
    reports=[]
    for threshold in THRESHOLDS:
        m=evaluate(labels,predictions(raw,threshold),'validation')
        reports.append(dict(threshold=threshold,metrics=m,nonlocalized_outputs=nonlocalized(m),
                            eligible=nonlocalized(m)<=nonlocalized(bm)))
    eligible=[r for r in reports if r['eligible']]
    winner=max(eligible,key=lambda r:(r['metrics']['counts'].get('localized_tp_16px',0),
                                     -r['metrics']['counts']['pred_visible'],r['threshold']))
    result=dict(selected_threshold=winner['threshold'],baseline=bm,selected=winner,
                validation_grid=reports,protocol_sha256=sha(OUT/'protocol.json'),
                raw_sha256=sha(OUT/'validation_raw.csv'))
    save('selection.json',result)
    print(json.dumps(dict(threshold=winner['threshold'],baseline=bm,selected=winner),indent=2),flush=True)


def evaluate_test():
    selection=json.loads((OUT/'selection.json').read_text())
    assert sha(OUT/'protocol.json')==selection['protocol_sha256']
    assert sha(LABELS)==json.loads((OUT/'protocol.json').read_text())['label_sha256']
    labels=[r for r in read(LABELS) if r['evaluation_split']=='test']
    raw=read(OUT/'test_raw.csv');pred=predictions(raw,selection['selected_threshold']);base=baseline(labels)
    bm=evaluate(labels,base,'test'); tm=evaluate(labels,pred,'test')
    improve=tm['counts'].get('localized_tp_16px',0)>bm['counts']['localized_tp_16px']
    safe=nonlocalized(tm)<=nonlocalized(bm)
    def correct(r,p):
        return (r['truth_visibility']=='visible' and int(p['visible'])==1 and
                math.hypot(float(p['x_px'])-float(r['truth_x_px']),
                           float(p['y_px'])-float(r['truth_y_px']))<=16)
    paired=dict(both_correct=0,only_baseline_correct=0,only_totnet_correct=0,neither_correct_visible=0)
    for r in labels:
        if r['truth_visibility']!='visible':continue
        cid=r['candidate_id'];t=int(r['local_frame'])
        b=correct(r,base[cid][t]);p=correct(r,pred[cid][t])
        key='both_correct' if b and p else 'only_baseline_correct' if b else 'only_totnet_correct' if p else 'neither_correct_visible'
        paired[key]+=1
    per_clip={cid:dict(baseline=evaluate([r for r in labels if r['candidate_id']==cid],base,'test'),
                       totnet=evaluate([r for r in labels if r['candidate_id']==cid],pred,'test')) for cid in sorted(pred)}
    result=dict(baseline=bm,totnet=tm,selected_threshold=selection['selected_threshold'],paired_visible=paired,per_clip=per_clip,
                diagnostic_ungated_argmax=evaluate(labels,predictions(raw,0.),'test'),
                diagnostic_note='Ungated argmax describes localization capacity only; it does not select or change the frozen visibility threshold.',
                selection_sha256=sha(OUT/'selection.json'),provisional_adoption_criterion_met=improve and safe,
                warning='Same-match reused test and unverified annotation provenance; not independent external accuracy.')
    save('test_results.json',result)
    details=[]
    for r in labels:
        cid=r['candidate_id'];t=int(r['local_frame']);p=pred[cid][t];b=base[cid][t]
        details.append(dict(r,totnet_x_px=p['x_px'],totnet_y_px=p['y_px'],totnet_visible=p['visible'],
                            totnet_peak=p['peak'],baseline_x_px=b['x_px'],baseline_y_px=b['y_px'],baseline_visible=b['visible']))
    with (OUT/'test_frame_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    print(json.dumps({k:v for k,v in result.items() if k!='per_clip'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('step',choices=['prepare','infer-validation','correct-origin-convention','select','infer-test','evaluate'])
    args=parser.parse_args()
    if args.step=='prepare':prepare()
    elif args.step=='infer-validation':infer('validation')
    elif args.step=='correct-origin-convention':correct_origin()
    elif args.step=='select':select()
    elif args.step=='infer-test':infer('test')
    else:evaluate_test()
