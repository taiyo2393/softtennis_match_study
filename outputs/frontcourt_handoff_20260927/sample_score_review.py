"""Freeze a stratified, outcome-blind validation sample for score suggestions."""
import csv
import random
from collections import defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent
with (OUT/'score_change_suggestions.csv').open(encoding='utf-8-sig',newline='') as file:
    scores=list(csv.DictReader(file))
with (OUT/'rally_label_review.csv').open(encoding='utf-8-sig',newline='') as file:
    rallies={r['candidate_id']:r for r in csv.DictReader(file)}

groups=defaultdict(list)
for score in scores:
    status=score['legacy_window_status']
    positive=score['suggestion_reason']=='one_point_increment_consistent'
    groups[(status,positive)].append(score)

rng=random.Random(20260927)
sample=[]
for key in sorted(groups):
    for score in sorted(rng.sample(groups[key],min(4,len(groups[key]))),
                        key=lambda r:r['candidate_id']):
        rally=rallies[score['candidate_id']]
        suffix=score['candidate_id'][-3:]
        sample.append(dict(candidate_id=score['candidate_id'],
                           sampling_seed=20260927,
                           stratum=f'{key[0]}__suggestion_{int(key[1])}',
                           stratum_population_n=len(groups[key]),
                           stratum_sample_n=min(4,len(groups[key])),
                           inverse_sampling_weight=f'{len(groups[key])/min(4,len(groups[key])):.3f}',
                           candidate_start_sec=rally['candidate_start_sec'],
                           candidate_end_sec=rally['candidate_end_sec'],
                           suggestion_reason=score['suggestion_reason'],
                           suggested_score_row=score['suggested_scoreboard_row'],
                           score_before_end=score['score_before_end'],
                           score_after_1s=score['score_after_1s'],
                           clip_file=f'{suffix}_score_validation_h264.mp4',
                           actual_score_change_row='',
                           score_change_linked_to_candidate_point='',
                           winner_team='',winner_camera_side='',
                           review_status='unreviewed'))
with (OUT/'score_suggestion_validation_sample.csv').open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(sample[0]))
    writer.writeheader()
    writer.writerows(sample)
print('sampled',len(sample),'of',len(scores))
for key in sorted(groups):
    print(key,len(groups[key]),'sampled',min(4,len(groups[key])))
