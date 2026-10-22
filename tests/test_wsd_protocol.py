import json
from pathlib import Path
import pytest
from src.wsd_protocol import wsd_factor,decide_wsd

RULE=json.loads((Path(__file__).parents[1]/'configs/convergence_decision_review2.json').read_text())


def data():
    stable={m:{s:{d:10.+i for d in RULE['domains']} for s in [27500,30000,47500,50000]} for i,m in enumerate(RULE['methods'])}
    cooled={m:{s:{d:10.+i for d in RULE['domains']} for s in [10000,20000,30000,50000]} for i,m in enumerate(RULE['methods'])}
    return stable,cooled


def test_wsd_does_not_decay_stable_tail_and_cooldown_uses_global_step():
    assert wsd_factor(50)==.5 and wsd_factor(100)==1
    assert wsd_factor(27500)==wsd_factor(50000)==1
    assert wsd_factor(11250,fork_step=10000)==.5
    assert wsd_factor(12500,fork_step=10000)==0
    with pytest.raises(ValueError):wsd_factor(10000,fork_step=10000)


def test_cooled_budget_includes_decay_and_complete_successor():
    stable,cooled=data();result=decide_wsd(stable,cooled,RULE)
    assert result['total_budget']==12500 and result['formal_grid_locked']
    del cooled['V'][20000]
    assert decide_wsd(stable,cooled,RULE)['status']=='awaiting_complete_curves'


def test_raw_or_cooled_improvement_extends_at_base_rate_and_cap_stays_locked():
    stable,cooled=data();stable['V'][27500]['nyuv2']+=.31
    del cooled['Xn'][50000]
    assert decide_wsd(stable,cooled,RULE)['status']=='awaiting_extended_curves'
    stable,cooled=data();cooled['V'][20000]['arb']+=.31
    assert decide_wsd(stable,cooled,RULE)['extension_required']
    stable['R'][47500]['hypersim']+=.31
    assert decide_wsd(stable,cooled,RULE)['status']=='inconclusive_still_improving'
