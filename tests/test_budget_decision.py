import json
from pathlib import Path
from src.budget_decision import decide_budget

RULE=json.loads((Path(__file__).parents[1]/'configs/convergence_decision_v2.json').read_text())


def curves(end=30000):
    return {m:{s:{d:10.+i for d in RULE['domains']} for s in range(2500,end+1,2500)}
            for i,m in enumerate(RULE['methods'])}


def test_earliest_shared_budget_and_incomplete_grid_lock():
    c=curves();assert decide_budget(c,RULE)['budget']==2500
    del c['V'][17500]
    assert decide_budget(c,RULE)['status']=='awaiting_complete_curves'


def test_any_domain_extension_and_capped_continuation():
    c=curves();c['V'][27500]['arb']+=.31
    result=decide_budget(c,RULE)
    assert result['extend_to']==50000 and result['status']=='awaiting_extended_curves'
    c=curves(50000);c['V'][27500]['arb']+=.31
    assert decide_budget(c,RULE)['reference_endpoint']==50000
    c['Xn'][47500]['nyuv2']+=.31
    assert decide_budget(c,RULE)['status']=='inconclusive_still_improving'


def test_exact_threshold_does_not_extend_and_nan_never_unlocks():
    c=curves();c['R'][27500]['hypersim']+=.3
    assert decide_budget(c,RULE)['status']=='budget_candidate'
    c['R'][12500]['arb']=float('nan')
    assert decide_budget(c,RULE)['formal_grid_locked']
    assert decide_budget(c,RULE)['status']=='awaiting_complete_curves'


def test_rank_flip_rejects_early_point_and_no_successor_is_not_invented():
    c=curves()
    c['V'][2500]['arb']=10.005;c['R'][2500]['arb']=10.0
    c['V'][5000]['arb']=9.99;c['R'][5000]['arb']=10.02
    result=decide_budget(c,RULE);assert result['budget']>=7500
    c=curves()
    for m in c:
        for s in range(2500,30000,2500):
            for d in RULE['domains']:c[m][s][d]+=1.
    # No extension: the last 2.5k improves by zero, but earlier points stay far away.
    for m in c:
        for d in RULE['domains']:c[m][27500][d]-=1.
    # A small rank reversal at the endpoint, with no >0.3 improvement trigger.
    c['R'][27500]['arb']=10.;c['V'][27500]['arb']=10.1
    c['R'][30000]['arb']=10.15;c['V'][30000]['arb']=10.
    assert decide_budget(c,RULE)['status']=='inconclusive_no_qualifying_budget'
