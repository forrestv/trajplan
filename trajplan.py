from __future__ import division

import bisect

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
    def simple2(cls, points, degree, **kwargs):
        # produces a spline that starts and ends at the first and last points,
        # at which it has C^(degree-1) continuity to a constant
        points = list(points)
        start, end = points[0], points[-1]
        points = points[1:-1]
        DUP = degree-1
        KNOTDUP = degree
        assert DUP >= 1
        knots = [0] * KNOTDUP + map(float, numpy.linspace(0, 1, 2*DUP + len(points) + degree + 1 - 2 * KNOTDUP)) + [1] * KNOTDUP
        points = [start]*DUP + points + [end]*DUP
        return cls(points, knots, degree, **kwargs)

import numpy
from matplotlib import pyplot, animation
a = lambda *x: numpy.array(list(x))
import math

N = 20
points = [a(math.cos(i/N*math.pi), math.sin(i/N*math.pi))+numpy.random.randn(2)*.04 for i in xrange(N+1)]

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

bs = BSpline.simple2(points, 2, lerp=mylerp)

if 0:
    px = [bs.evaluate(x)['p'] for x in numpy.linspace(0, 1, 1000)]

    def animate(i):
        pyplot.cla()
        pyplot.plot(*zip(*px))
        import time
        t = time.time() % 10 / 10
        res = bs.evaluate(t)
        pyplot.scatter(*zip(*points))
        pyplot.scatter(*zip(*[res['p']]) + [80])
        pyplot.arrow(*(list(res['p']) + list(.1*res['v'])))
        pyplot.arrow(*(list(res['p']) + list(.003*res['a'])))

    fig = pyplot.figure()

    ani = animation.FuncAnimation(fig, animate, interval=25)

    pyplot.show()

else:
    inf = 1e1000
    
    def w(p, v):
        return -v
    m = 1
    
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
        
        # u = p + d2s/dt2 v
        p = m * res['a'] * ds_over_dt**2 - w(res['p'], res['v'] * ds_over_dt)
        v = m * res['v']
        
        #print p, v
        
        allowed = (-inf, inf)
        for halfspace_normal, halfspace_dist in [
            (numpy.array([1, 0]), -1),
            (numpy.array([-1, 0]),-1),
            (numpy.array([0, 1]), -1),
            (numpy.array([0, -1]),-1),
        ]:
            this_allowed = line_halfspace_intersection(p, v, halfspace_normal, halfspace_dist)
            #print this_allowed
            allowed = max(allowed[0], this_allowed[0]), min(allowed[1], this_allowed[1])
        return allowed
    
    range_is_valid = lambda (lo, hi): lo <= hi
    
    def find_maximum_ds_over_dt(s):
        assert range_is_valid(get_allowable_d2s_over_dt2_range(s, 0))
        a = 1
        while range_is_valid(get_allowable_d2s_over_dt2_range(s, a)):
            a *= 2
        # a is invalid
        # breakpoint is in (0, a]
        breakpoint_range = 0, a
        for i in xrange(20):
            if range_is_valid(get_allowable_d2s_over_dt2_range(s, (breakpoint_range[0] + breakpoint_range[1])/2)):
                breakpoint_range = (breakpoint_range[0] + breakpoint_range[1])/2, breakpoint_range[1]
            else:
                breakpoint_range = breakpoint_range[0], (breakpoint_range[0] + breakpoint_range[1])/2
        #print breakpoint_range
        return breakpoint_range[0]
    
    def advance((s, ds_over_dt), d2s_over_dt2, ds):
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
    px = []
    px1 = []
    px2 = []
    px3 = []
    ds_over_dt = 0
    for s, s2 in zip(numpy.linspace(0, 1, N)[:-1], numpy.linspace(0, 1, N)[1:]):
        rng = get_allowable_d2s_over_dt2_range(s, ds_over_dt)
        ds = 1/(N-1)
        print s, ds_over_dt, can_stop_from(advance((s, ds_over_dt), rng[1], ds), ds)
        px.append((s, ds_over_dt))
        px1.append((s, rng[0]))
        px2.append((s, rng[1]))
        if not range_is_valid(rng): break
        if can_stop_from(advance((s, ds_over_dt), rng[1], ds), ds):
            chosen = rng[1]
        else:
            assert can_stop_from(advance((s, ds_over_dt), rng[0], ds), ds)
            chosen = rng[0]
        px2.append((s, chosen))
        _, ds_over_dt = advance((s, ds_over_dt), chosen, ds)
    
    pyplot.plot(*zip(*px))
    pyplot.plot(*zip(*px1))
    pyplot.plot(*zip(*px2))
    pyplot.plot(*zip(*px3))
    px = [(x, find_maximum_ds_over_dt(x)) for x in numpy.linspace(0, 1, N)]
    pyplot.plot(*zip(*px))
    pyplot.show()
    fdasfdsa
    
    px = [(x, numpy.linalg.norm(bs.evaluate(x)['v'])) for x in numpy.linspace(0, 1, N)]
    pyplot.plot(*zip(*px))
    
    px = [(x, find_maximum_ds_over_dt(x)) for x in numpy.linspace(0, 1, N)]
    pyplot.plot(*zip(*px))
    
    px = [(x, get_allowable_d2s_over_dt2_range(x, 0)[0]) for x in numpy.linspace(0, 1, N)]
    pyplot.plot(*zip(*px))
    px = [(x, get_allowable_d2s_over_dt2_range(x, 0)[1]) for x in numpy.linspace(0, 1, N)]
    pyplot.plot(*zip(*px))
    
    px = [(x, get_allowable_d2s_over_dt2_range(x, find_maximum_ds_over_dt(x))[0]) for x in numpy.linspace(0, 1, N)]
    pyplot.plot(*zip(*px))
    px = [(x, get_allowable_d2s_over_dt2_range(x, find_maximum_ds_over_dt(x))[1]) for x in numpy.linspace(0, 1, N)]
    pyplot.plot(*zip(*px))
    
    pyplot.show()
