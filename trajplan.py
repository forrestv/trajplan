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

def w(p, v):
    return -v

px = [bs.evaluate(x)['p'] for x in numpy.linspace(0, 1, 1000)]

def animate(i):
    pyplot.cla()
    pyplot.plot(*zip(*px))
    import time
    t = time.time() % 10 / 10
    res = bs.evaluate(t)
    pyplot.scatter(*zip(*[res['p']]))
    pyplot.arrow(*(list(res['p']) + list(.1*res['v'])))
    pyplot.arrow(*(list(res['p']) + list(.003*res['a'])))

fig = pyplot.figure()

ani = animation.FuncAnimation(fig, animate, interval=25)

pyplot.show()
