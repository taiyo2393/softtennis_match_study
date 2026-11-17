"""Export candidate-level appearance examples from existing explicit reviews.

All cases belong to one rally and cannot support an independent train/test split.
Missing or unreviewed candidates never become negatives.
"""
import json,math
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2,np

OUT=ROOT/'ball_044_reviewed_appearance_cases'

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def crop(im,x,y,size):
    left=round(x)-size//2;top=round(y)-size//2
    assert left>=0 and top>=0 and left+size<=im.shape[1] and top+size<=im.shape[0]
    return im[top:top+size,left:left+size].copy(),left,top

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'manifest.json').exists()
    registry=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    bp=Path(registry['reviewed_output']);ball=read(bp);assert sha(bp)==registry['reviewed_output_sha256']
    protected={str(p):sha(p) for p in [ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_tracking_bundles.json',ROOT/'reviewed_person_overrides.json',bp]}
    reviews={}
    for name in registry['review_records']:
        path=Path(name);r=load(path);reviews[r['review_id']]=(r,path);protected[str(path)]=sha(path)
    cases=[]
    for t in registry['confirmed_ball_candidate_local_frames']:
        r=ball[t];assert int(r['visible']) and r['ball_candidate_review']=='user_confirmed_is_ball'
        review,path=reviews[r['human_review_id']]
        explicit_frames=[int(i) for i in review.get('confirmed_ball_candidate_local_frames',[])]
        explicit_frames += [int(c['local_frame']) for c in review.get('candidates',[]) if c.get('candidate_is_ball') is True]
        assert t in explicit_frames,(t,path)
        view='tiled2x' if 'tiled' in r['position_source'] else 'full_frame_component'
        cases.append(dict(case_id=f'{view}_{t}_real',local_frame=t,label='confirmed_ball_enclosed',
            x_px=float(r['x_px']),y_px=float(r['y_px']),component_peak=r['component_peak'],inference_view=view,
            human_review_id=r['human_review_id'],review_record=str(path.resolve())))
    for review,path in reviews.values():
        for r in review.get('rejected_raw_observations',[]):
            t=int(r['local_frame']);x,y=r['rejected_xy_px']
            cases.append(dict(case_id=f'full_frame_raw_{t}_false',local_frame=t,label='confirmed_not_ball_enclosed',
                x_px=x,y_px=y,component_peak='',inference_view='full_frame_raw',
                human_review_id=review['review_id'],review_record=str(path.resolve())))
        for r in review.get('rejected_tiled_components',[]):
            t=int(r['local_frame'])
            cases.append(dict(case_id=f'tiled2x_{t}_{r["display_label"]}_false',local_frame=t,label='confirmed_not_ball_enclosed',
                x_px=float(r['x_px']),y_px=float(r['y_px']),component_peak=r['peak'],inference_view='tiled2x',
                human_review_id=review['review_id'],review_record=str(path.resolve())))
    assert len(cases)==28 and len({r['case_id'] for r in cases})==28
    cases.sort(key=lambda c:(c['local_frame'],c['label'],c['inference_view']))
    for dirname in ['patch96','context224']:(OUT/dirname).mkdir(exist_ok=True)
    rows=[]
    for c in cases:
        t=c['local_frame'];image=SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'/f'{t:06d}.jpg'
        im=cv2.imread(str(image));assert im is not None and im.shape[:2]==(1080,1920)
        row=dict(c,source_frame=114635+t,rally_id='match1_rally_window_044',match_id='legacy_match1',
            split_group='legacy_match1:rally044',center_precision_groundtruth=0,physical_visibility_label='not_created',
            original_image=str(image),original_image_sha256=sha(image))
        for size,dirname in [(96,'patch96'),(224,'context224')]:
            patch,left,top=crop(im,c['x_px'],c['y_px'],size);p=OUT/dirname/(c['case_id']+'.png')
            assert cv2.imwrite(str(p),patch)
            decoded=cv2.imread(str(p));assert np.array_equal(decoded,patch)
            row.update({f'{dirname}_path':str(p.resolve()),f'{dirname}_sha256':sha(p),f'{dirname}_left':left,f'{dirname}_top':top})
        rows.append(row)
    write(OUT/'cases.csv',rows)
    # Four existing labels demonstrate why appearance should be checked, not assigned by motion.
    ids=['tiled2x_1089_real','tiled2x_1091_real','tiled2x_1091_A_false','tiled2x_1092_A_false']
    atlas=np.full((744,1344,3),22,np.uint8)
    for k,identity in enumerate(ids):
        c=next(r for r in rows if r['case_id']==identity)
        tile=cv2.resize(cv2.imread(c['context224_path']),(672,672),interpolation=cv2.INTER_NEAREST)
        # Each panel uses the same crop size and3x enlargement; show original candidate center.
        cv2.circle(tile,(336,336),18,(80,255,100) if c['label']=='confirmed_ball_enclosed' else (0,165,255),2)
        # Arrange four panels across one wide strip, then scale entire strip for convenient export.
        small=cv2.resize(tile,(336,336),interpolation=cv2.INTER_AREA)
        atlas[72:408,k*336:(k+1)*336]=small
        cv2.putText(atlas,identity.replace('tiled2x_',''),(k*336+8,28),0,.52,(245,245,245),1,cv2.LINE_AA)
        cv2.putText(atlas,'reviewed BALL' if c['label']=='confirmed_ball_enclosed' else 'reviewed FALSE',(k*336+8,54),0,.54,(245,245,245),1,cv2.LINE_AA)
    # The second row presents native96px patches, enlarged3x with consistent scale.
    for k,identity in enumerate(ids):
        c=next(r for r in rows if r['case_id']==identity)
        patch=cv2.resize(cv2.imread(c['patch96_path']),(288,288),interpolation=cv2.INTER_NEAREST)
        atlas[444:732,k*336+24:k*336+312]=patch
    cv2.imwrite(str(OUT/'reviewed_counterexamples.png'),atlas)
    assert all(sha(Path(p))==h for p,h in protected.items())
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    manifest=dict(cases=len(rows),confirmed_ball_enclosed=18,confirmed_not_ball_enclosed=10,
        source_rallies=1,match_count=1,independent_test_split_possible=False,
        no_new_human_labels_requested=True,unreviewed_candidates_not_used_as_negatives=True,
        missing_frames_not_used_as_negatives=True,source_images_unchanged=True,reviewed_inputs_unchanged=True,
        patches='Native96x96 and224x224 RGB image content exported as lossless PNG; no learned enhancement or interpolation to create ball positions',
        center_meaning='Reviewed marker centers, not independently labelled precise ball centers',
        limitation='Adjacent frames and shared rally are correlated; do not randomly split these rows for reported evaluation. Same frame may have both a real and a false candidate.',
        labels_table_sha256=sha(OUT/'cases.csv'),input_hashes=protected,
        automatic_classifier_trained=False,independent_accuracy_established=False)
    dump(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='input_hashes'},indent=2))

if __name__=='__main__':main()
