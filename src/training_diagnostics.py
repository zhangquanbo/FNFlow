"""Sample-count-weighted time bins and every-update clipping diagnostics."""
import math


def learning_rate_factor(step,schedule_steps):
    """Original warmup/cosine, then its floor for a preregistered continuation."""
    warmup=min(100,schedule_steps//10)
    return min(1.,step/max(1,warmup))*(.1+.9*.5*(1+math.cos(math.pi*min(step,schedule_steps)/schedule_steps)))


def resume_segment_action(completed,steps,phase,extend_to,curve):
    if completed<=steps:return 'run'
    if phase=='convergence' and not extend_to and any(p['update']==steps for p in curve):
        return 'skip_initial'  # Preserve a partially completed continuation; never relabel its weights as 30k.
    raise RuntimeError('Resume checkpoint exceeds requested segment; explicit continuation required')


class TimeLossBins:
    def __init__(self, edges):
        self.edges=list(edges)
        if any(b<=a for a,b in zip(edges,edges[1:])):raise ValueError('Unordered time edges')
        self.rows=[{'samples':0,**{key:0. for key in ['mse','unit','cosine','loss']}} for _ in edges[:-1]]

    def add(self, times, values):
        for j,t in enumerate(times):
            if not self.edges[0]<=t<self.edges[-1]:raise ValueError('Time outside diagnostic bins')
            i=next(i for i,(a,b) in enumerate(zip(self.edges,self.edges[1:])) if a<=t<b)
            self.rows[i]['samples']+=1
            for key in ['mse','unit','cosine','loss']:
                value=float(values[key][j])
                if not math.isfinite(value):raise ValueError('Nonfinite time-bin loss')
                self.rows[i][key]+=value

    def summary(self):
        return [{'t_min':a,'t_max':b,'samples':row['samples'],
                 **{key:row[key]/row['samples'] if row['samples'] else None for key in ['mse','unit','cosine','loss']}}
                for a,b,row in zip(self.edges,self.edges[1:],self.rows)]
