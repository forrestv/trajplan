from __future__ import division

import bisect
import math

import numpy
from matplotlib import pyplot, animation

inf = 1e1000; assert math.isinf(inf)


class BSpline(object):
    def __init__(self, points, knots, degree, lerp=lambda a, b, x, dxdt: (1-x)*a + x*b):
        assert len(knots) == len(points) + degree + 1
        points = list(points)
        knots = map(float, knots)
        self.points = points
        self.knots = knots
        self.degree = degree
        self.lerp = lerp
    
    def evaluate(self, x):
        assert x >= self.knots[0] and x <= self.knots[-1]
        if x <= self.knots[0]+1e-6: x = self.knots[0]+1e-6
        if x >= self.knots[-1]-1e-6: x = self.knots[-1]-1e-6
        
        l = bisect.bisect_right(self.knots, x) - 1
        if l == len(self.knots)-1: l -= 1
        assert l >= 0
        assert self.knots[l] <= x < self.knots[l+1]
        
        n = self.degree
        c = [self.knots[i] for i in xrange(l-n+1, l+n+1)]
        d = [self.points[i] for i in xrange(l-n, l+1)]
        for k in xrange(n):
            d = [self.lerp(d[i], d[i+1], (x - c[i+k])/(c[i+n] - c[i+k]), 1/(c[i+n] - c[i+k])) for i in xrange(n-k)]
        assert len(d) == 1
        return d[0]
    
    @classmethod
    def simple(cls, points, degree, **kwargs):
        # produces a spline that starts and ends at the first and last points,
        # at which it has C^degree continuity to a constant
        DUP = degree-1
        KNOTDUP = degree
        assert DUP >= 0
        knots = [0] * KNOTDUP + map(float, numpy.linspace(0, 1, 2*DUP + len(points) + degree + 1 - 2 * KNOTDUP)) + [1] * KNOTDUP
        points = [points[0]]*DUP + points + [points[-1]]*DUP
        return cls(points, knots, degree, **kwargs)
    
    @classmethod
    def simple_nostop(cls, points, degree, **kwargs):
        # produces a spline that starts and ends at the first and last points,
        # at which it has only C^1 continuity to a constant
        DUP = degree-1
        KNOTDUP = degree
        assert DUP >= 1
        knots = [0] * KNOTDUP + map(float, numpy.linspace(0, 1, 2*DUP + (len(points) - (2*(degree-1))) + degree + 1 - 2 * KNOTDUP)) + [1] * KNOTDUP
        return cls(points, knots, degree, **kwargs)

spline_control_point_count = 20
spline_control_points = [numpy.array([
    math.cos(i/spline_control_point_count*math.pi),
    math.sin(i/spline_control_point_count*math.pi),
])+numpy.random.randn(2)*.04 for i in xrange(spline_control_point_count+1)]

def mylerp(a, b, x, dxdt):
    if not isinstance(a, dict):
        a = dict(
            p=a,
            v=numpy.array([0, 0]),
            a=numpy.array([0, 0]),
        )
        b = dict(
            p=b,
            v=numpy.array([0, 0]),
            a=numpy.array([0, 0]),
        )
    return dict(
        p=(1-x)*a['p'] + x*b['p'],
        v=((1-x)*a['v'] + x*b['v']) + (b['p'] - a['p'])*dxdt,
        a=((1-x)*a['a'] + x*b['a']) + (b['v'] - a['v'])*dxdt + (b['v'] - a['v'])*dxdt,
    )

bs = BSpline.simple_nostop(spline_control_points, 2, lerp=mylerp)

def w(p, v):
    return -v

m = 1

u_constraints = [ # halfspace_normal, halfspace_dist; see line_halfspace_intersection
    (numpy.array([1, 0]), -1),
    (numpy.array([-1, 0]),-1),
    (numpy.array([0, 1]), -1),
    (numpy.array([0, -1]),-1),
]

range_is_valid = lambda (lo, hi): lo <= hi
intersect_ranges = lambda (lo1, hi1), (lo2, hi2): (max(lo1, lo2), min(hi1, hi2))

def line_halfspace_intersection(line_start, line_dir, halfspace_normal, halfspace_dist):
    # halfspace_normal * allowed_point >= halfspace_dist
    # line is defined by line_start + x * line_dir
    # returns interval of x that is allowed
    
    # halfspace_normal . p >= halfspace_dist
    # halfspace_normal . (line_start + x * line_dir) >= halfspace_dist
    # halfspace_normal . line_start + x * halfspace_normal . line_dir >= halfspace_dist
    # x * (halfspace_normal . line_dir) >= halfspace_dist - halfspace_normal . line_start
    if halfspace_normal.dot(line_dir) > 0:
        return ((halfspace_dist - halfspace_normal.dot(line_start)) / halfspace_normal.dot(line_dir), inf)
    elif halfspace_normal.dot(line_dir) < 0:
        return (-inf, (halfspace_dist - halfspace_normal.dot(line_start)) / halfspace_normal.dot(line_dir))
    else: # line is parallel to halfspace boundary
        if halfspace_normal.dot(line_start) >= halfspace_dist: # everything allowed
            return (-inf, inf)
        else: # nothing allowed
            return (inf, -inf)

def get_allowable_d2s_over_dt2_range(s, ds_over_dt):
    res = bs.evaluate(s)
    
    # u = line_start + d2s/dt2 line_dir
    line_start = m * res['a'] * ds_over_dt**2 - w(res['p'], res['v'] * ds_over_dt)
    line_dir = m * res['v']
    
    allowed = (-inf, inf)
    for halfspace_normal, halfspace_dist in u_constraints:
        this_allowed = line_halfspace_intersection(line_start, line_dir, halfspace_normal, halfspace_dist)
        allowed = intersect_ranges(allowed, this_allowed)
    return allowed

def find_maximum_ds_over_dt(s):
    assert range_is_valid(get_allowable_d2s_over_dt2_range(s, 0))
    a = 1
    while range_is_valid(get_allowable_d2s_over_dt2_range(s, a)):
        a *= 2
    # a is invalid
    # breakpoint is in (0, a]; binary search to find it
    breakpoint_range = 0, a
    for i in xrange(20):
        if range_is_valid(get_allowable_d2s_over_dt2_range(s, (breakpoint_range[0] + breakpoint_range[1])/2)):
            breakpoint_range = (breakpoint_range[0] + breakpoint_range[1])/2, breakpoint_range[1]
        else:
            breakpoint_range = breakpoint_range[0], (breakpoint_range[0] + breakpoint_range[1])/2
    #print breakpoint_range
    return breakpoint_range[0]

def advance((s, ds_over_dt), d2s_over_dt2, ds):
    # advance
    if ds_over_dt**2 + 2 * ds * d2s_over_dt2 <= 0:
        ds_over_dt = 0
    else:
        ds_over_dt = math.sqrt(ds_over_dt**2 + 2 * ds * d2s_over_dt2)
    return s + ds, ds_over_dt

def can_stop_from((s, ds_over_dt), ds):
    while True:
        if ds_over_dt == 0: return True
        if s >= 1: return False
        rng = get_allowable_d2s_over_dt2_range(s, ds_over_dt)
        if not range_is_valid(rng): return False
        s, ds_over_dt = advance((s, ds_over_dt), rng[0], ds)

N = 1001

px3 = []
ds_over_dt = 0
s = 0
ds = 1/(N-1)
res = []
while True:
    res.append((s, ds_over_dt))
    print s, ds_over_dt,
    if s != 0 and ds_over_dt == 0:
        print
        break
    rng = get_allowable_d2s_over_dt2_range(s, ds_over_dt)
    print can_stop_from(advance((s, ds_over_dt), rng[1], ds), ds)
    if not range_is_valid(rng): break
    if can_stop_from(advance((s, ds_over_dt), rng[1], ds), ds):
        chosen = rng[1]
    else:
        assert can_stop_from(advance((s, ds_over_dt), rng[0], ds), ds)
        chosen = rng[0]
    px3.append((s+ds/2, chosen))
    s, ds_over_dt = advance((s, ds_over_dt), chosen, ds)

t = 0
ts = [(0, t)]
for (s1, ds_over_dt1), (s2, ds_over_dt2) in zip(res[:-1], res[1:]):
    # circular integration..!
    # find new_t such that a circle that goes (s1, t) and (s2, new_t)
    # while respecting the slope conditions at each point.
    # avoids divide by zero that arises from standard integration.
    # consequences of inventing new math: unknown.
    t = t + (s2 - s1) * (1 - ds_over_dt1*ds_over_dt2 + math.sqrt((1+ds_over_dt1**2)*(1+ds_over_dt2**2)))/(ds_over_dt1+ds_over_dt2)
    ts.append((s2, t))

res2 = []
s = 0
last_s = None
for i in xrange(N):
    if s > 1:
        s2 = 1
        assert i == N-1
    else:
        s2 = s
    speval = bs.evaluate(s2)
    assert ts[i][0] == s
    t = ts[i][1]
    assert res[i][0] == s
    ds_over_dt = res[i][1]
    if i == N-1:
        d2s_over_dt2 = 0
    else:
        assert px3[i][0] == s+ds/2
        d2s_over_dt2 = px3[i][1]
    p = speval['p']
    v = speval['v'] * ds_over_dt
    a = speval['a'] * ds_over_dt**2 + speval['v'] * d2s_over_dt2
    res2.append(dict(
        s=s,
        t=t,
        ds_over_dt=ds_over_dt,
        d2s_over_dt2=d2s_over_dt2,
        p=p,
        v=v,
        a=a,
        u=m*a - w(p, v),
    ))
    last_s = s
    s += ds

if 0:
    pyplot.plot(*zip(*[(x['t'], x['u'][0]) for x in res2]))
    pyplot.plot(*zip(*[(x['t'], x['u'][1]) for x in res2]))
    pyplot.show()
else:
    spline_sampled_points = [bs.evaluate(x)['p'] for x in numpy.linspace(0, 1, 1000)]
    def animate(i):
        pyplot.cla()
        pyplot.gca().set_aspect('equal')
        pyplot.plot(*zip(*spline_sampled_points))
        pyplot.scatter(*zip(*spline_control_points))
        import time
        t = time.time() % res2[-1]['t']
        i = min(xrange(N), key=lambda i: abs(res2[i]['t'] - t))
        inst = res2[i]
        pyplot.scatter(*zip(*[inst['p']]) + [80])
        #pyplot.arrow(*(list(inst['p']) + list(.1*inst['v'])))
        pyplot.arrow(*(list(inst['p']) + list(.1*inst['u'])))
        pyplot.plot(*zip(*[inst['p'] + .1*numpy.array(x) for x in [(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]]))

    fig = pyplot.figure()

    ani = animation.FuncAnimation(fig, animate, interval=25)

    pyplot.show()
