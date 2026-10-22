"""Sample-count-weighted time bins and every-update clipping diagnostics."""
import math


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
