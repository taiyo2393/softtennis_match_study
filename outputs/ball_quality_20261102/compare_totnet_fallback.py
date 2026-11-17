"""Freeze missing-only fusion on validation before any test inference."""
import argparse
import csv
import json

from compare_totnet import (OUT, LABELS, THRESHOLDS, read, evaluate, baseline,
                           predictions, nonlocalized, sha)


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def fuse(base,net):
    return {cid:{t:dict(b,position_source='baseline_model_observed') if int(b['visible']) else
                     dict(net[cid][t],position_source='totnet_observed_when_baseline_missing' if int(net[cid][t]['visible']) else 'missing')
                 for t,b in byframe.items() if t in net[cid]}
            for cid,byframe in base.items()}


def select():
    assert not (OUT/'test_raw.csv').exists(),'Freeze before test inference'
    assert not (OUT/'fallback_selection.json').exists()
    protocol=dict(trigger='TOTNet standalone validation TP71 vs baseline80 prompted a missing-only candidate before test inference',
                  rule='Retain every visible baseline coordinate. Only when baseline missing, use origin-aware TOTNet above fixed threshold.',
                  thresholds=THRESHOLDS,selection='Maximize localized validation TP with no additional nonlocalized labeled outputs; ties fewer outputs then higher threshold',
                  adoption='Development candidate only if test TP increases without added nonlocalized outputs. Continuous-video review needed before replacing active output.',
                  label_sha256=sha(LABELS),totnet_protocol_sha256=sha(OUT/'protocol.json'))
    save('fallback_protocol.json',protocol)
    labels=[r for r in read(LABELS) if r['evaluation_split']=='validation']
    base=baseline(labels);bm=evaluate(labels,base,'validation');raw=read(OUT/'validation_raw.csv')
    candidates=[]
    for threshold in THRESHOLDS:
        m=evaluate(labels,fuse(base,predictions(raw,threshold)),'validation')
        candidates.append(dict(threshold=threshold,metrics=m,nonlocalized_outputs=nonlocalized(m),eligible=nonlocalized(m)<=nonlocalized(bm)))
    winner=max([c for c in candidates if c['eligible']],key=lambda c:(c['metrics']['counts'].get('localized_tp_16px',0),-c['metrics']['counts']['pred_visible'],c['threshold']))
    result=dict(selected_threshold=winner['threshold'],baseline=bm,selected=winner,validation_grid=candidates,
                protocol_sha256=sha(OUT/'fallback_protocol.json'))
    save('fallback_selection.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='validation_grid'},indent=2),flush=True)


def test():
    choice=json.loads((OUT/'fallback_selection.json').read_text())
    protocol=json.loads((OUT/'fallback_protocol.json').read_text())
    assert sha(OUT/'fallback_protocol.json')==choice['protocol_sha256']
    assert sha(LABELS)==protocol['label_sha256']
    assert sha(OUT/'protocol.json')==protocol['totnet_protocol_sha256']
    labels=[r for r in read(LABELS) if r['evaluation_split']=='test']
    base=baseline(labels);raw=read(OUT/'test_raw.csv');net=predictions(raw,choice['selected_threshold']);fusion=fuse(base,net)
    bm=evaluate(labels,base,'test');fm=evaluate(labels,fusion,'test')
    for cid,byframe in fusion.items():
        for t,p in byframe.items():
            b=base[cid][t]
            if int(b['visible']):assert all(str(p[k])==str(b[k]) for k in ('visible','x_px','y_px'))
            elif int(p['visible']):assert all(p[k]==net[cid][t][k] for k in ('x_px','y_px','visible'))
    result=dict(baseline=bm,fallback=fm,selected_threshold=choice['selected_threshold'],
                selection_sha256=sha(OUT/'fallback_selection.json'),baseline_visible_coordinates_preserved=True,
                added_positions_come_from_totnet=True,
                development_candidate_criterion_met=(fm['counts'].get('localized_tp_16px',0)>bm['counts']['localized_tp_16px'] and nonlocalized(fm)<=nonlocalized(bm)),
                warning='Only 150 sampled test frames; same-match reused test with unverified annotation provenance. No full-video adoption or external accuracy claim.')
    save('fallback_test_results.json',result)
    rows=[dict(p,candidate_id=cid,local_frame=t) for cid,byframe in fusion.items() for t,p in byframe.items()]
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'fallback_test_sample_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select','test']);args=parser.parse_args()
    select() if args.phase=='select' else test()
