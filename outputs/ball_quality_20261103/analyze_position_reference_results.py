"""Independent numerical check and bounded diagnosis of completed reference pilot."""
import math,itertools,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump

STAGE=BASE/'position_reference_evaluation_20261110_r55'
BUNDLE=BASE/'identity_position_bundle'

def main():
    labels=json.loads((STAGE/'human_reference_labels.json').read_text(encoding='utf-8'));manifest=json.loads((STAGE/'private_sampling_manifest.json').read_text(encoding='utf-8'));items={r['image_id']:r for r in manifest['items']};pred=defaultdict(list)
    for r in read(BUNDLE/'position_policy_observations.csv'):
        if r['analysis_proxy_present']=='1':pred[r['candidate_id'][-3:],r['source_frame'],r['camera_side'],r['position_policy']].append(r)
    raw=read(STAGE/'evaluation/per_context_side_results.csv');points=[];checked=0
    for row in raw:
        item=items[row['image_id']];side=row['camera_side'];refs=[dict(slot=side+'_'+lr,**labels['labels'][row['image_id']][side+'_'+lr]) for lr in ['left','right']];rs=[r for r in refs if r['state']=='point'];ps=pred[row['tag'],row['source_frame'],side,row['position_policy']]
        assert len(rs)==int(row['reference_position_points']) and len(ps)==int(row['prediction_points'])
        if row['evaluation_state']!='scored':assert row['mean_point_error_px']=='';continue
        assert len(rs)==len(ps)==2;choices=[]
        for order in [(0,1),(1,0)]:
            dist=[math.hypot(float(ps[order[i]]['analysis_proxy_x'])-rs[i]['x'],float(ps[order[i]]['analysis_proxy_y'])-rs[i]['y']) for i in range(2)];choices.append((sum(dist),order,dist))
        total,order,dist=min(choices,key=lambda x:x[0]);assert math.isclose(total/2,float(row['mean_point_error_px']),abs_tol=1e-10);assert math.isclose(total/2/2160,float(row['mean_point_error_over_image_height']),abs_tol=1e-12)
        for i,d in enumerate(dist):
            p=ps[order[i]];dx=float(p['analysis_proxy_x'])-rs[i]['x'];dy=float(p['analysis_proxy_y'])-rs[i]['y'];points.append(dict(sampling_group=row['sampling_group'],image_id=row['image_id'],tag=row['tag'],source_frame=row['source_frame'],camera_side=side,position_policy=row['position_policy'],reference_slot=rs[i]['slot'],native_player_slot=p['player_slot'],native_track_id=p['native_track_id'],reference_x=rs[i]['x'],reference_y=rs[i]['y'],native_proxy_x=p['analysis_proxy_x'],native_proxy_y=p['analysis_proxy_y'],dx_px=dx,dy_px=dy,distance_px=d,body_warning=p['body_overlap_warning'],not_independent_ID_truth=1));checked+=1
        xy=np.array([[r['x'],r['y']] for r in rs]);pp=np.array([[float(p['analysis_proxy_x']),float(p['analysis_proxy_y'])] for p in ps]);expected=dict(centroid_x_error_norm=abs(xy[:,0].mean()-pp[:,0].mean())/3840,centroid_y_error_norm=abs(xy[:,1].mean()-pp[:,1].mean())/2160,x_gap_error_norm=abs(np.ptp(xy[:,0])-np.ptp(pp[:,0]))/3840,y_gap_error_norm=abs(np.ptp(xy[:,1])-np.ptp(pp[:,1]))/2160)
        for k,v in expected.items():assert math.isclose(float(row[k]),v,abs_tol=1e-12);checked+=1
    write(STAGE/'per_matched_reference_point_diagnostic.csv',points);uniform=[r for r in points if r['sampling_group']=='uniform_one_per_context' and r['position_policy']=='retain_native_proxy'];assert len(uniform)==34
    side_stats=[]
    for side in ['near','far']:
        selected=[r for r in uniform if r['camera_side']==side];side_stats.append(dict(camera_side=side,scored_points=len(selected),source_contexts=len({r['tag'] for r in selected}),conditional_mean_distance_px=float(np.mean([r['distance_px'] for r in selected])),conditional_median_distance_px=float(np.median([r['distance_px'] for r in selected])),mean_dx_px=float(np.mean([r['dx_px'] for r in selected])),mean_dy_px=float(np.mean([r['dy_px'] for r in selected])),positive_dy_points=sum(r['dy_px']>0 for r in selected)))
    write(STAGE/'near_far_error_diagnostic.csv',side_stats)
    groups=[]
    for group in ['uniform_one_per_context','known_body_warning_challenge']:
        for policy in ['retain_native_proxy','withhold_body_warning_proxy']:
            selected=[r for r in raw if r['sampling_group']==group and r['position_policy']==policy];groups.append(dict(sampling_group=group,position_policy=policy,pairs=len(selected),predicted_complete_pairs=sum(int(r['prediction_pair_available']) for r in selected),reference_complete_position_pairs=sum(int(r['position_pair_evaluable']) for r in selected),scored_pairs=sum(r['evaluation_state']=='scored' for r in selected),reference_unplaceable_pairs=sum(r['evaluation_state']=='reference_positions_unavailable' for r in selected),prediction_incomplete_pairs_regardless_reference=sum(r['prediction_pair_available']=='0' for r in selected)))
    write(STAGE/'all_denominators.csv',groups)
    # A conditional agreement measure is not an accuracy percentage or rejection success rate.
    result=dict(uniform_contexts=10,uniform_context_side_pairs=20,reference_position_pairs_available=18,scored_pairs=17,scored_points=34,scored_contexts=9,conditional_mean_point_distance_px=float(np.mean([r['distance_px'] for r in uniform])),conditional_median_point_distance_px=float(np.median([r['distance_px'] for r in uniform])),image_size=[3840,2160],uniform_error_equal_between_policies=True,uniform_complete_predictions_before=18,uniform_complete_predictions_after=17,missing_prediction_context006=True,known_warning_examples=4,known_warning_pairs_with_unplaceable_reference=3,known_warning_pairs_with_two_reference_points=1,only_evaluable_warning_pair_case='019',only_evaluable_warning_pair_frame=46396,only_evaluable_warning_pair_mean_distance_px=7.560126491683436,that_pair_becomes_incomplete_under_mask=True,known_warning_sample_not_a_representative_accuracy_rate=True,side_diagnostics=side_stats,independently_checked_numeric_values=checked)
    dump(STAGE/'analysis_result.json',result)
    # Scientific figure: conditional error per sampled context, no zero-error imputation for missing data.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,2,figsize=(12.6,5.0),gridspec_kw={'width_ratios':[1.25,1]});tags=sorted({r['tag'] for r in raw if r['sampling_group']=='uniform_one_per_context'});colors={'near':'#2563a6','far':'#dc7e1d'}
    for side,offset in [('near',-.12),('far',.12)]:
        rows=[r for r in raw if r['sampling_group']=='uniform_one_per_context' and r['position_policy']=='retain_native_proxy' and r['camera_side']==side]
        for r in rows:
            y=tags.index(r['tag'])+offset
            if r['evaluation_state']=='scored':axes[0].scatter(float(r['mean_point_error_px']),y,color=colors[side],s=38,label=side if r['tag']==tags[0] else None)
            else:axes[0].text(58,y,'no prediction / partial ref' if r['prediction_points']=='0' and r['reference_position_points']!='2' else 'no prediction' if r['prediction_points']=='0' else 'reference unavailable',va='center',fontsize=8,color=colors[side])
    axes[0].set(yticks=range(len(tags)),yticklabels=tags,xlabel='Mean image-point distance per pair (native pixels)',ylabel='Sampled context',title='Uniform sample: 17 scored pairs out of 20',xlim=(0,108));axes[0].invert_yaxis();axes[0].grid(axis='x',alpha=.2);axes[0].legend(frameon=False)
    for i,side in enumerate(['near','far']):
        selected=[r for r in uniform if r['camera_side']==side];dx=[r['dx_px'] for r in selected];dy=[r['dy_px'] for r in selected];axes[1].scatter(dx,dy,s=32,alpha=.8,color=colors[side],label=f'{side} ({len(selected)} points)')
    axes[1].axhline(0,color='#555',lw=.8);axes[1].axvline(0,color='#555',lw=.8);axes[1].set(xlabel='Native proxy x minus manual reference x (px)',ylabel='Native proxy y minus manual reference y (px)',title='Signed difference: positive y points downward');axes[1].grid(alpha=.15);axes[1].legend(frameon=False);fig.suptitle('Same-match development pilot; manual image references, not surveyed ground contact',fontsize=11);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(STAGE/'position_reference_diagnostic.png',dpi=170);fig.savefig(STAGE/'position_reference_diagnostic.pdf');plt.close(fig)
    # Display worst near/far agreements plus the known-mask counterexample and an unplaceable pair.
    selection=[]
    for side in ['near','far']:selection.append((max([r for r in uniform if r['camera_side']==side],key=lambda r:r['distance_px']),'Largest conditional point distance in uniform '+side))
    target=next(r for r in points if r['sampling_group']=='known_body_warning_challenge' and r['position_policy']=='retain_native_proxy' and r['image_id']=='P012' and r['native_player_slot']=='far_rear');selection.append((target,'Reference placed; warning would remove this native proxy'))
    tiles=[];details=[]
    for index,(r,reason) in enumerate(selection,1):
        image=cv2.imread(str(BUNDLE/'reference_pilot/images'/(r['image_id']+'.png')));native=next(p for p in pred[r['tag'],r['source_frame'],r['camera_side'],'retain_native_proxy'] if p['player_slot']==r['native_player_slot']);x1,y1,x2,y2=[float(native[k]) for k in ['x1','y1','x2','y2']];l=max(0,int(x1)-120);t=max(0,int(y1)-80);rr=min(3840,int(x2)+120);b=min(2160,int(y2)+120);proxy=(round(float(r['native_proxy_x'])),round(float(r['native_proxy_y'])));ref=(round(r['reference_x']),round(r['reference_y']));cv2.rectangle(image,(round(x1),round(y1)),(round(x2),round(y2)),(220,220,220),2);cv2.line(image,proxy,ref,(255,255,255),2);cv2.circle(image,proxy,8,(0,160,255),-1);cv2.drawMarker(image,ref,(255,255,0),cv2.MARKER_TILTED_CROSS,20,3);crop=image[t:b,l:rr];scale=min(700/crop.shape[1],500/crop.shape[0]);small=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)));canvas=np.zeros((590,700,3),np.uint8);xx=(700-small.shape[1])//2;yy=90+(500-small.shape[0])//2;canvas[yy:yy+small.shape[0],xx:xx+small.shape[1]]=small;cv2.putText(canvas,f'{r["tag"]} F{r["source_frame"]} {r["camera_side"]}: {r["distance_px"]:.1f}px',(12,25),0,.65,(255,255,255),1);cv2.putText(canvas,'Orange: native bbox-bottom; cyan X: manual reference',(12,51),0,.48,(255,255,255),1);cv2.putText(canvas,reason,(12,77),0,.45,(255,255,255),1);file=STAGE/f'diagnostic_example_{index:02d}.jpg';assert cv2.imwrite(str(file),canvas);tiles.append(canvas);details.append(dict(reason=reason,point=r,image=file.name,image_sha256=sha(file),crop_native_xyxy=[l,t,rr,b],aspect_preserved=True))
    assert cv2.imwrite(str(STAGE/'diagnostic_examples.jpg'),np.hstack(tiles));dump(STAGE/'diagnostic_selection.json',dict(items=details,selection_after_fixed_primary_evaluation=True,not_representative_accuracy_sampling=True,reference_points_not_modified=True));print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
