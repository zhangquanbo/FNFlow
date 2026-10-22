"""Deterministic preregistered budget selection; incomplete curves never pass."""
import itertools
import math


def decide_budget(curves, rule):
    methods,domains=rule['methods'],rule['domains'];interval=rule['evaluation_interval']
    first=rule['initial_endpoint'];extension=rule['extended_endpoint'];eps=rule.get('numerical_epsilon_deg',1e-9)
    def value(m,s,d):
        x=float(curves[m][s][d])
        if not math.isfinite(x):raise ValueError('Nonfinite curve point')
        return x
    def improvements(end):
        return {m+'/'+d:value(m,end-interval,d)-value(m,end,d) for m in methods for d in domains}
    try:
        tail=improvements(first)
        extend=any(v>rule['extension_threshold_deg']+eps for v in tail.values())
        endpoint=extension if extend else first
        if extend:
            end_tail=improvements(extension)
            if any(v>rule['extension_threshold_deg']+eps for v in end_tail.values()):
                return {'status':'inconclusive_still_improving','extend_to':extension,'formal_grid_locked':True}
        def rank(step,domain):
            return tuple(0 if abs(delta)<=rule['ranking_tie_tolerance_deg']+eps else (1 if delta>0 else -1)
                for a,b in itertools.combinations(methods,2) for delta in [value(a,step,domain)-value(b,step,domain)])
        for step in range(interval,endpoint,interval):
            # Require a complete shared grid, even if an early B might otherwise pass.
            for m in methods:
                for d in domains:value(m,step,d)
        for step in range(interval,endpoint,interval):
            close=all(abs(value(m,step,d)-value(m,endpoint,d))<=rule['endpoint_tolerance_deg']+eps
                      for m in methods for d in domains)
            stable=all(rank(step,d)==rank(step+interval,d) for d in domains)
            if close and stable:
                return {'status':'budget_candidate','budget':step,'reference_endpoint':endpoint,
                        'extension_triggered':extend,'tail_improvements_deg':tail,'formal_grid_locked':True,
                        'note':'Candidate only; common clipping review and a separate formal configuration still required'}
        return {'status':'inconclusive_no_qualifying_budget','reference_endpoint':endpoint,'formal_grid_locked':True}
    except (KeyError,TypeError,ValueError):
        # Direction to extend is useful even while 50k data are incomplete.
        return {'status':'awaiting_extended_curves' if 'extend' in locals() and extend else 'awaiting_complete_curves',
                'extend_to':extension if 'extend' in locals() and extend else None,'formal_grid_locked':True}
