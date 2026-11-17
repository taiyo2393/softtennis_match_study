"""Resume frozen trial after six-case source lacked sample_id. NO refitting."""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18,ResNet18_Weights
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from compare_frozen_ball_backbones import predict,WEIGHTS
from fit_frozen_ball_appearance_probe import features as handcrafted
from prepare_training_appearance_review import crop96
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err

OUT=ROOT/'ball_training_only_appearance_comparison';BASE=ROOT/'ball_temporal_path_comparison'
METHODS=['handcrafted672','frozen_resnet18_512']
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    assert not (OUT/'decision.json').exists()
    protocol=load(OUT/'protocol.json');training=load(OUT/'training_result.json')
    assert sha(ROOT/'compare_training_only_ball_appearance.py')==protocol['script_sha256']
    assert all(sha(Path(p))==h for p,h in protocol['protected_input_hashes'].items())
    heads={}
    for method in METHODS:
        p=OUT/f'final_{method}.npz';assert sha(p)==training['final_model_hashes'][method];heads[method]=dict(np.load(p))
    result=dict(training_clip_holdout=training['internal_clip_holdout'])
    for role in ['validation','challenge']:
        original=read(BASE/f'{role}_selected.csv');bm=metrics(original);result[role]={};fs=np.load(OUT/f'{role}_features.npz')
        for method in METHODS:
            pred=read(OUT/f'{role}_{method}.csv');pm=metrics(pred);lost=sorted(good_ids(original)-good_ids(pred))
            scores=predict(heads[method],fs[method]);lookup=dict(zip(fs['sample_ids'],scores));assert len(original)==len(pred)
            for a,b in zip(original,pred):
                assert a['sample_id']==b['sample_id']
                if int(a['visible']):
                    assert abs(float(b['appearance_score'])-float(lookup[a['sample_id']]))<1e-12
                    assert int(b['visible'])==int(lookup[a['sample_id']]>=.5)
                else:assert int(b['visible'])==0
                if int(b['visible']):assert (a['x_px'],a['y_px'])==(b['x_px'],b['y_px'])
            result[role][method]=dict(baseline=bm,proposed=pm,correct_lost=lost,nonregression=not lost and err(pm)<=err(bm),
                withheld=sum(int(a['visible']) and not int(b['visible']) for a,b in zip(original,pred)))
    # Only fix missing metadata; no learned parameters, images, or label rules changed.
    old=read(ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015/balltrack_predictions_proposed.csv')
    false=read(ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv')
    train=read(OUT/'training_manifest.csv');train_cids={r['candidate_id'] for r in train};train_hashes={r['source_image_sha256'] for r in train}
    folder=OUT/'six_false_patches';folder.mkdir(exist_ok=True);patches=[];meta=[]
    for i,r in enumerate(false):
        t=int(r['local_frame']);b=old[t];cid=b['candidate_id'];assert cid not in train_cids
        assert r['candidate_is_ball']=='0' and int(b['visible'])
        assert (float(b['x_px']),float(b['y_px']))==(float(r['displayed_x_px']),float(r['displayed_y_px']))
        p=SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg';assert sha(p) not in train_hashes
        im=cv2.imread(str(p));assert im.shape==(1080,1920,3);patch=crop96(im,float(b['x_px']),float(b['y_px']));path=folder/f'{i:04d}.png'
        if path.exists():assert np.array_equal(cv2.imread(str(path)),patch)
        else:assert cv2.imwrite(str(path),patch)
        meta.append(dict(sample_id=f'{cid}_frame_{t:06d}',candidate_id=cid,local_frame=t,source_frame=b['source_frame'],
            source_image=str(p),source_image_sha256=sha(p),crop_x=b['x_px'],crop_y=b['y_px'],raw_patch96=str(path),patch_sha256=sha(path)))
        patches.append(patch)
    write(OUT/'six_false_patch_manifest.csv',meta)
    backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));backbone.fc=torch.nn.Identity()
    backbone.eval().requires_grad_(False).to('cuda');transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    images=[transform(torch.from_numpy(p[32:64,32:64,::-1].copy()).permute(2,0,1)) for p in patches]
    with torch.inference_mode():deep=backbone(torch.stack(images).to('cuda')).cpu().numpy().astype(np.float64)
    fs=dict(handcrafted672=np.stack([handcrafted(Path(r['raw_patch96'])) for r in meta]),frozen_resnet18_512=deep)
    np.savez_compressed(OUT/'six_false_features.npz',sample_ids=np.array([r['sample_id'] for r in meta]),**fs)
    result['six_false']={};outcomes=[]
    for method in METHODS:
        scores=predict(heads[method],fs[method]);result['six_false'][method]=dict(total=6,withheld=int((scores<.5).sum()),retained=int((scores>=.5).sum()))
        for r,s in zip(meta,scores):outcomes.append(dict(method=method,local_frame=r['local_frame'],x_px=r['crop_x'],y_px=r['crop_y'],score=float(s),withheld=int(s<.5),physical_visibility='not_labelled',true_center='unknown'))
    write(OUT/'six_false_predictions.csv',outcomes)
    eligible=[m for m in METHODS if result['validation'][m]['nonregression'] and result['challenge'][m]['nonregression'] and
        (err(result['validation'][m]['proposed'])<2 or result['six_false'][m]['withheld']>0)]
    assert not eligible,'A method passed; complete its preplanned audit before closing.'
    assert all(sha(Path(p))==h for p,h in protocol['protected_input_hashes'].items())
    assert all(sha(OUT/f'final_{m}.npz')==training['final_model_hashes'][m] for m in METHODS)
    result.update(status='fixed_training_only_appearance_comparison_complete_not_adopted',audit=None,audit_eligible_methods=eligible,
        automatic_adoption=False,operational_outputs_changed=False,protected_inputs_unchanged=True,physical_absence_labels_added=0,interpolation=0,
        independent_generalization_established=False,resumed_without_refitting=True)
    dump(OUT/'resume_record.json',dict(cause='Six-case source rows lack sample_id; initial run stopped after training and validation/challenge prediction files were saved.',
        fix='Construct sample_id from source clip/local frame for six-case metadata only. Reuse frozen heads/features; verify validation/challenge scores exactly.',
        no_training_rerun=True,no_parameter_changes=True,no_threshold_changes=True,no_changed_evaluation_predictions=True,
        script_sha256=sha(Path(__file__)),original_script_sha256=sha(ROOT/'compare_training_only_ball_appearance.py')))
    dump(OUT/'decision.json',result)
    for role in ['validation','challenge']:
        for m,r in result[role].items():print(role,m,r['baseline']['counts'],r['proposed']['counts'],'lost',len(r['correct_lost']))
    print('six_false',result['six_false']);print('internal',result['training_clip_holdout'])

if __name__=='__main__':main()
