"""Strictly load official tennis weights and verify one real-video forward pass.

This does not evaluate accuracy. Keep the released source unmodified. The
checkpoint's EasyDict metadata is mapped to an inert local dict class, using
weights_only=True rather than arbitrary pickle loading.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'third_party/python_deps'))
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102'))
from ball_alignment import SOURCE


def main():
    repo=ROOT/'third_party/TOTNet'
    checkpoint=ROOT/'model_weights/totnet_tennis_best.pth'
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    assert set(unsafe)=={'easydict.EasyDict'},unsafe
    with torch.serialization.safe_globals([(OrderedDict,'easydict.EasyDict')]):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    weights=state['state_dict']
    weights={k.removeprefix('module.'):v for k,v in weights.items()}
    path=repo/'src/model/TOTNet.py'
    spec=importlib.util.spec_from_file_location('official_totnet_core',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    channels=weights['block1.conv_layers.0.block.0.weight'].shape[0]
    model=module.TemporalConvNet(input_shape=(288,512),spatial_channels=channels,num_frames=5)
    result=model.load_state_dict(weights,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device).eval()
    # Follow Tennis_Dataset + Normalize in TRAINING code: cv2 BGR input.
    # The released folder demo uses RGB, so log this discrepancy explicitly.
    base=SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_001/frames'
    target=112
    images=[]
    for frame in range(target-4,target+1):
        im=cv2.imread(str(base/f'{frame:06d}.jpg'));assert im is not None
        im=cv2.resize(im,(512,288)).astype(np.float32)/255.
        im=(im-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        images.append(np.moveaxis(im,-1,0))
    tensor=torch.from_numpy(np.stack(images)[None]).to(device)
    with torch.inference_mode():output=model(tensor)
    assert output.shape==(1,288*512) and torch.isfinite(output).all()
    assert abs(float(output.sum())-1.)<.001
    index=int(output.argmax());y,x=divmod(index,512)
    configs={}
    for name,value in state.items():
        if name in ('configs','config','args') and isinstance(value,dict):
            attrs=dict(value);attrs.update(vars(value))
            configs[name]={k:str(v) for k,v in attrs.items() if k in ('num_frames','num_channels','bidirect','dataset_choice','img_size','model_choice')}
    report=dict(repository='https://github.com/AugustRushG/TOTNet',
        revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_metadata=configs,strict_weights_loaded=True,weight_tensors=len(weights),
        channels=int(channels),device=device,input_shape=list(tensor.shape),output_shape=list(output.shape),
        softmax_sum=float(output.sum()),target_local_frame=target,input_local_frames=list(range(target-4,target+1)),
        example_argmax_1920x1080=[x*1920/512,y*1080/288],
        input_color='BGR following training dataset; demo has RGB inconsistency',
        status='strict_load_and_real_frame_forward_pass_only_not_accuracy_evaluation',
        limitation='Peak softmax is not a calibrated ball visibility probability; target/color preprocessing must be fixed before comparison.')
    (ROOT/'totnet_setup_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
