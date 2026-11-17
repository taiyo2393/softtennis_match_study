from conservative_pair_tracker import ConservativePairTracker
from identity_guard import old
from test_identity_guard import scene


def close_uniforms(frame,missing=False,new_id=105):
    rows,features = scene(frame,missing=missing,new_id=new_id)
    # Similar uniforms, but enough local feature difference that the earlier
    # margin-only guard could accept a long reidentification.
    indices = {int(r['track_id']):j for j,r in enumerate(rows)}
    partner = indices[4]
    features[partner] = .8*features[partner]*0+.8*scene(0)[1][2]+.2*scene(0)[1][3]
    return rows,features


def main():
    t = ConservativePairTracker()
    for f in range(8):
        t.step(f,*close_uniforms(f))
    assert t.similar_pairs['far']['not_distinct_by_color']
    for f in range(8,120):
        t.step(f,*close_uniforms(f,missing=True))
    for f in range(120,140):
        output = t.step(f,*close_uniforms(f))
        assert not any(i == 2 for i,_,_,_,_ in output)
    assert t.states['far_rear']['frame'] == 7
    assert not any(e['event'] == 'reacquired_candidate' for e in t.events)
    assert any(e['event'] == 'reacquisition_withheld_similar_pair' for e in t.events)
    # A clear short handover of the same local trajectory can be retained.
    short = ConservativePairTracker()
    for f in range(8):
        short.step(f,*close_uniforms(f))
    rows,features = close_uniforms(8)
    rows[2]['track_id'] = 105
    assert any(i == 2 for i,_,_,_,_ in short.step(8,rows,features))
    print('Passed: long reidentification of similar uniforms withheld despite local feature margin; state rollback; successful-recovery event removed; clear short handover preserved.')


if __name__ == '__main__':
    main()
