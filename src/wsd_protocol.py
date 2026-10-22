"""Review2 learning-rate schedule and deterministic cooled-budget decisions."""
import itertools,math


def wsd_factor(step,warmup=100,fork_step=None,cooldown=2500):
    if fork_step is None:return min(1.,step/max(1,warmup))
    if step<=fork_step or step>fork_step+cooldown:raise ValueError('Update outside cooldown branch')
    return max(0.,1-(step-fork_step)/cooldown)


def decide_wsd(stable,cooled,rule):
    methods,domains=rule['methods'],rule['domains'];eps=rule['numerical_epsilon_deg']
    threshold=rule['extension_threshold_deg'];initial=rule['initial_fork_steps']
    def value(data,m,s,d):
        result=float(data[m][s][d])
        if not math.isfinite(result):raise ValueError('Nonfinite result')
        return result
    def gain(data,a,b):return {m+'/'+d:value(data,m,a,d)-value(data,m,b,d) for m in methods for d in domains}
    try:
        raw=gain(stable,27500,30000);after=gain(cooled,20000,30000)
        extend=any(v>threshold+eps for v in [*raw.values(),*after.values()])
        forks=initial+[50000] if extend else initial
        if extend:
            tail=gain(stable,47500,50000);cooled_tail=gain(cooled,30000,50000)
            if any(v>threshold+eps for v in [*tail.values(),*cooled_tail.values()]):
                return {'status':'inconclusive_still_improving','extension_required':True,'formal_grid_locked':True}
        # Require every planned cooled result, not a favourable subset.
        for m in methods:
            for step in forks:
                for d in domains:value(cooled,m,step,d)
        def rank(step,d):
            return tuple(0 if abs(delta)<=rule['ranking_tie_tolerance_deg']+eps else (1 if delta>0 else -1)
                for a,b in itertools.combinations(methods,2) for delta in [value(cooled,a,step,d)-value(cooled,b,step,d)])
        for step,next_step in zip(forks,forks[1:]):
            close=all(abs(value(cooled,m,step,d)-value(cooled,m,forks[-1],d))<=rule['endpoint_tolerance_deg']+eps for m in methods for d in domains)
            if close and all(rank(step,d)==rank(next_step,d) for d in domains):
                return {'status':'budget_candidate','stable_updates':step,'cooldown_updates':rule['cooldown_updates'],
                        'total_budget':step+rule['cooldown_updates'],'reference_stable_endpoint':forks[-1],
                        'extension_required':extend,'formal_grid_locked':True}
        return {'status':'inconclusive_no_qualifying_budget','extension_required':extend,'formal_grid_locked':True}
    except (KeyError,TypeError,ValueError):
        return {'status':'awaiting_extended_curves' if 'extend' in locals() and extend else 'awaiting_complete_curves',
                'extension_required':bool(locals().get('extend',False)),'formal_grid_locked':True}
