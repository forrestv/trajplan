from __future__ import division

import bisect

class BSpline(object):
    def __init__(self, points, knots, degree):
        assert len(knots) == len(points) + degree + 1
        points = list(points)
        knots = map(float, knots)
        #assert len(points) == len(knots)
        self.points = points
        self.knots = knots
        self.degree = degree
        #for i in xrange(self.degree):
        #    self.points.insert(0, self.points[0])
        #    self.knots.insert(0, self.knots[0])
        #    self.points.append(self.points[-1])
        #    self.knots.append(self.knots[-1])
    
    def evaluate(self, x):
        assert x >= self.knots[0] and x <= self.knots[-1]
        if x <= self.knots[0]+1e-6: x = self.knots[0]+1e-6
        if x >= self.knots[-1]-1e-6: x = self.knots[-1]-1e-6
        
        l = bisect.bisect_right(self.knots, x) - 1
        if l == len(self.knots)-1: l -= 1
        assert l >= 0
        assert self.knots[l] <= x < self.knots[l+1]
        
        n = self.degree
        print x, l, (l-n+1, l+n), (l-n, l)
        c = [self.knots[i] for i in xrange(l-n+1, l+n+1)]
        d = [self.points[i] for i in xrange(l-n, l+1)]
        lerp = lambda a, b, x: (1-x)*a + x*b
        for k in xrange(n):
            d = [lerp(d[i], d[i+1], (x - c[i+k])/(c[i+n] - c[i+k])) for i in xrange(n-k)]
        assert len(d) == 1
        return d[0]


import numpy
from matplotlib import pyplot
bs = BSpline([0, 0, 0, 6, 0, 0, 0], [-2, -2, -2, -2, -1, 0, 1, 2, 2, 2, 2], 3)
pyplot.plot(*zip(*[(x, bs.evaluate(x)) for x in numpy.linspace(-2, 2, 100)]))
#pyplot.figure(2)
#for deg in [0, 1, 2]: #xrange(10):
#    bs = BSpline([1, 2, 3, 6, 5, 6, 7], numpy.linspace(0, 1, 7), deg)
#    pyplot.plot(*zip(*[(x, bs.evaluate(x)) for x in numpy.linspace(0, 1, 1000)]))
pyplot.show()
