"""Development refinement: corroborate upper-body veto with the same exemplar."""
from far_identity_guard import original,solve,part_distance,gate as first_gate,PARAMETERS as FIRST_PARAMETERS

PARAMETERS=dict(FIRST_PARAMETERS,upper_negative_requires_same_exemplar_whole_body_support=True,
    upper_negative_whole_body_slack=FIRST_PARAMETERS['negative_margin'])

def gate(feature,anchors,background):
    allowed,diagnostic=first_gate(feature,anchors,background,upper=False)
    competitors=background+anchors['near_front']+anchors['near_rear']
    witnesses=[]
    for i,slot in enumerate(['far_rear','far_front']):
        witness=[]
        for exemplar in competitors:
            whole=original.a.app_distance(exemplar,feature);upper=part_distance(exemplar,feature,0)
            if (upper+PARAMETERS['upper_body_negative_margin']<diagnostic['own_upper'][i]
                    and whole<=diagnostic['own'][i]+PARAMETERS['upper_negative_whole_body_slack']):
                witness.append((whole,upper))
        if witness:allowed[i]=False
        witnesses.append(witness)
    return allowed,dict(diagnostic,corroborating_negative_counts=[len(w) for w in witnesses])
