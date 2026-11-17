"""Record user-confirmed manual center provenance and fixed clip partitions."""
import json
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha

OUT=ROOT/'balltrack_manual_finetune_pilot'
TRAIN=[1,3,4,5,6,7,9,10]
VALIDATION=[11,12,13,14,15,17,19,20]
AUDIT=[2,8,16,18]

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'data_manifest.json').exists()
    path=SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv';labels=read(path);assert len(labels)==300
    assert sha(path)==load(ROOT/'ball_sequence_candidate_supervision/legacy_label_audit.json')['source_sha256']
    provenance=dict(recorded_date='2026-11-03',source_label_table=str(path),source_sha256=sha(path),
        question='以前の20動画区間×15フレーム＝300件の球位置注釈は、どのように付けたものですか？',
        user_answer='自分で球をクリックして付けた',annotator_kind='human_user',
        scope='User confirms the existing ball-position annotation was made by manually clicking the ball. Preserve original visibility states and coordinates.',
        actual_annotation_date='not_established',blindness_to_original_predictions='not_established',
        interpretation='Manual center annotations can supply supervised development targets. This does not turn repeatedly used same-match data into an independent external test.',
        reviewer_id_field='Blank source fields are retained; no invented author ID or annotation date is written to the source table.')
    dump(OUT/'manual_annotation_provenance.json',provenance)
    inventory={r['ball_pilot_id']:r for r in read(ROOT/'play_start_audit/ball_pilot_start_inventory.csv')}
    sets=dict(train=TRAIN,validation=VALIDATION,audit=AUDIT);assigned=[];input_hashes={str(path):sha(path)}
    assert set(TRAIN)|set(VALIDATION)|set(AUDIT)==set(range(1,21))
    assert not (set(TRAIN)&set(VALIDATION) or set(TRAIN)&set(AUDIT) or set(VALIDATION)&set(AUDIT))
    for role,numbers in sets.items():
        selected=[]
        for r in labels:
            if int(r['candidate_id'].rsplit('_',1)[1]) not in numbers:continue
            t=int(r['local_frame']);cid=r['candidate_id'];base=SOURCE/'data/balltrack/racketvision/match1'/cid
            assert r['truth_visibility'] in ('visible','not_visible') and t>=3
            assert int(r['source_frame'])==int(inventory[cid]['clip_first_source_frame'])+t
            if r['truth_visibility']=='visible':assert 0<float(r['truth_x_px'])<1920 and 0<float(r['truth_y_px'])<1080
            else:assert not r['truth_x_px'] and not r['truth_y_px']
            frames=[base/'frames'/f'{j:06d}.jpg' for j in range(t-3,t+1)]
            median=base/'runs/threshold030_median120/median.npz'
            assert all(p.exists() for p in frames) and median.exists()
            q=dict(r,finetune_role=role,split_group='legacy_match1:'+cid,
                input_frames_json=json.dumps([str(p) for p in frames]),input_frame_sha256_json=json.dumps([sha(p) for p in frames]),
                median_path=str(median),median_sha256=sha(median),input_ends_at_labelled_frame=1)
            selected.append(q);assigned.append(q)
        assert len(selected)==15*len(numbers)
        write(OUT/f'{role}_labels.csv',selected)
    assert len(assigned)==300 and len({r['sample_id'] for r in assigned})==300
    for a,b in [('train','validation'),('train','audit'),('validation','audit')]:
        ah={h for r in assigned if r['finetune_role']==a for h in json.loads(r['input_frame_sha256_json'])}
        bh={h for r in assigned if r['finetune_role']==b for h in json.loads(r['input_frame_sha256_json'])}
        assert not ah&bh,(a,b)
    for p in [ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']:input_hashes[str(p)]=sha(p)
    manifest=dict(status='manual_center_source_confirmed_fixed_partitions_exported',source_sha256=sha(path),
        train_clips=TRAIN,validation_clips=VALIDATION,audit_clips=AUDIT,
        split_counts={role:dict(Counter(r['truth_visibility'] for r in assigned if r['finetune_role']==role)) for role in sets},
        roles='train120,validation120,audit60. Audit clips contain all36 previously reviewed candidates and never enter training or checkpoint selection.',
        phase_scope='Existing dataset includes preparation,play and transitions; do not call its aggregate metric in-play-only accuracy.',
        independence='Same-match, repeatedly seen development data; no claim of independent external performance.',
        no_random_frame_split=True,four_frame_input_hashes_disjoint_between_roles=True,
        source_labels_unchanged=True,input_hashes=input_hashes,
        partition_hashes={role:sha(OUT/f'{role}_labels.csv') for role in sets},
        annotation_provenance=str((OUT/'manual_annotation_provenance.json').resolve()))
    assert sha(path)==manifest['source_sha256'];dump(OUT/'data_manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='input_hashes'},indent=2))

if __name__=='__main__':main()
