"""Main cohort selected by user: original rear hitters H1/Hashiba, H2/Ueoka.

Existing annotations and 98-event analysis stay untouched. Derived files and
scientific figures are separated from the previous interim presentation.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'analysis/rear_hitter_main80'
TIMES = [-0.5, -0.3, -0.2, -0.1]
ALLOWED = {'H1':'後衛H1', 'H2':'後衛H2'}

def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:return list(csv.DictReader(f))

def write(name, rows):
    assert rows
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows, cohort, group):
    out=[]
    for t in TIMES:
        r=[x for x in rows if float(x['timepoint_sec'])==t]
        n=sum(x['movement_is_directional']=='1' for x in r)
        m=sum(x['direction_matches_actual']=='1' for x in r)
        assert m<=n<=len(r)
        out.append(dict(cohort=cohort,group=group,timepoint_sec=t,total_events=len(r),
                        directional_events=n,matches=m,disagreements=n-m,unclassified=len(r)-n,
                        agreement_percent=100*m/n if n else '',classification_percent=100*n/len(r) if r else ''))
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[BASE/'annotations/match1_hitter_identity_review.csv',
             BASE/'analysis/match1_court_plane_directional_motion_t0p015_event_timepoints.csv']
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    labels,features=map(read,sources)
    ids={r['event_id']:r for r in labels}
    assert len(ids)==len(labels)==98 and all(r['review_status']=='confirmed' for r in labels)
    assert {r['event_id'] for r in features}==set(ids)
    assert len({(r['event_id'],r['timepoint_sec']) for r in features})==len(features)==392
    for h in ALLOWED:assert len({r['hitter_description'] for r in labels if r['hitter_id']==h})==1
    cohort=[{**r,'main_analysis_included':int(r['hitter_id'] in ALLOWED),
             'selection_reason':'original_rear_hitter' if r['hitter_id'] in ALLOWED else 'other_physical_hitter_secondary_only'} for r in labels]
    assert sum(r['main_analysis_included'] for r in cohort)==80
    enriched=[]
    for r in features:
        a=ids[r['event_id']]
        assert r['target_front_player_id']==a['target_front_player_id'] and r['impact_source_frame']==a['impact_source_frame']
        enriched.append({**r,'hitter_id':a['hitter_id'],'hitter_name':'後衛'+a['hitter_id']})
    main_rows=[r for r in enriched if r['hitter_id'] in ALLOWED]
    secondary=[r for r in enriched if r['hitter_id'] not in ALLOWED]
    main_summary=summarize(main_rows,'main80','全体')
    pair_summary=[]
    for front,h in sorted({(r['target_front_player_id'],r['hitter_id']) for r in main_rows}):
        rows=[r for r in main_rows if r['target_front_player_id']==front and r['hitter_id']==h]
        pair_summary+=summarize(rows,'main80',f'前衛{front} 対 {ALLOWED[h]}')
    for total in main_summary:
        group=[r for r in pair_summary if r['timepoint_sec']==total['timepoint_sec']]
        for key in ['total_events','directional_events','matches']:assert total[key]==sum(r[key] for r in group)
    write('cohort_manifest_98.csv',cohort);write('main80_event_timepoints.csv',main_rows)
    write('secondary18_event_timepoints.csv',secondary);write('main80_summary.csv',main_summary)
    write('main80_pair_summary.csv',pair_summary);write('secondary18_summary.csv',summarize(secondary,'secondary18','全体'))
    common={t:{r['event_id'] for r in main_rows if float(r['timepoint_sec'])==t and r['movement_is_directional']=='1'} for t in TIMES}
    shared=common[-0.2]&common[-0.1]
    metadata={'source_sha256':hashes,'selection':'User-approved original rear hitters H1 and H2; H3/H4 retained separately.',
              'movement_threshold_court_width':0.015,'preceding_window_seconds':0.1,
              'main_unique_events':80,'secondary_unique_events':18,
              'common_directional_events_minus02_minus01':len(shared),
              'interpretation':'Directional agreement conditional on classifiable displacement; not poach success, conscious prediction accuracy, or optimal onset timing.'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    # Matplotlib is available in the existing research environment; no new packages.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/YuGothM.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/YuGothM.ttc').get_name(),
                         'font.size':16,'axes.unicode_minus':False,'svg.fonttype':'none'})
    def axes(ax):
        ax.set_xlim(-.55,-.05);ax.set_ylim(0,115);ax.set_yticks(range(0,101,20))
        ax.set_xticks(TIMES,[f'{abs(t):.1f}秒前' for t in TIMES]);ax.grid(axis='y',alpha=.25)
        ax.set_xlabel('相手後衛がボールを打つ時刻を基準とした評価時点',labelpad=12)
        ax.set_ylabel('方向の一致率（％）');ax.spines[['top','right']].set_visible(False)
    def draw(ax,rows,color):
        ax.plot(TIMES,[r['agreement_percent'] for r in rows],color=color,marker='o',lw=2.7,ms=8)
        for t,r in zip(TIMES,rows):
            ax.annotate(f"{r['agreement_percent']:.1f}％\n({r['matches']}/{r['directional_events']})",(t,r['agreement_percent']),
                        xytext=(0,-15 if r['agreement_percent']>=95 else 12),textcoords='offset points',ha='center',
                        va='top' if r['agreement_percent']>=95 else 'bottom',fontsize=17,color=color,
                        bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=220,facecolor='white');fig.savefig(OUT/(name+'.svg'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6.5));axes(ax);draw(ax,main_summary,'#214F76')
    ax.set_title('本来の後衛が打った80場面：全体の方向の一致率',pad=24,fontsize=21)
    fig.subplots_adjust(left=.11,right=.97,top=.83,bottom=.24)
    fig.text(.11,.035,'括弧内：一致数／左右を判定した場面数\n移動量の基準：コート幅の1.5％。各時点で集計対象は異なる。',fontsize=13)
    save(fig,'全体80場面_一致率')
    fig,axs=plt.subplots(1,2,figsize=(15,7))
    for ax,group,color in zip(axs,sorted({r['group'] for r in pair_summary}),['#214F76','#B45F06']):
        r=[r for r in pair_summary if r['group']==group];axes(ax);draw(ax,r,color)
        ax.set_title(f"{group}（{r[0]['total_events']}場面）",fontsize=21,pad=22)
        ax.set_xlabel('相手後衛の打球前の評価時点')
    fig.subplots_adjust(left=.065,right=.98,wspace=.25,top=.80,bottom=.25)
    fig.suptitle('前衛と相手後衛の組合せ別の方向の一致率',fontsize=24,y=.97)
    fig.text(.065,.04,'括弧内：一致数／左右を判定した場面数。基準はコート幅の1.5％。\n各時点で集計対象が異なり、組合せ間の差から前衛・後衛それぞれの影響は分離できない。',fontsize=14)
    save(fig,'組合せ別80場面_一致率')
    text=['# 主解析：本来の後衛本人が打った80場面','',
          '後衛H1が47場面、後衛H2が33場面。H3・H4の各9場面は副解析として保存。元の98場面のファイルは変更していない。','',
          '対象選択のみ変更。各時点の直前0.1秒の変位、閾値1.5％、本人視点の左右を維持。','',
          '|時点|判定数|一致数|一致率|','|---|---:|---:|---:|']
    for r in main_summary:text.append(f"|{r['timepoint_sec']}秒|{r['directional_events']}|{r['matches']}|{r['agreement_percent']:.1f}％|")
    text+=['',f'-0.2秒と-0.1秒で共通して左右を判定した場面は{len(shared)}件。','',
           '## 発表資料の更新箇所','',
           '- 対象数：98から80へ。P1は33場面、P2は47場面。',
           '- 全体・組合せ別グラフ、および説明原稿の分母・一致率を更新する。',
           '- 旧版の98場面の結果と混在させない。',
           '- 一致率最大の時点は、最適な動き出し時刻やポーチ成功率を意味しない。',
           '- 80場面は本人IDによる選別。全場面の打球位置やポーチ実施を新しく確認したものではない。']
    (OUT/'主解析80場面_結果と変更点.md').write_text('\n'.join(text),encoding='utf-8')
    for p in sources:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(BASE))]
    print(json.dumps({'summary':main_summary,'pair_summary':pair_summary,'shared':len(shared),'source_unchanged':True},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
