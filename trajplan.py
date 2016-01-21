from __future__ import division

import bisect

class CardinalBSpline(object):
    def __init__(self, points, control, degree):
        points = list(points)
        control = map(float, control)
        #assert len(points) == len(control)
        self.points = points
        self.control = control
        self.degree = degree
        #for i in xrange(self.degree):
        #    self.points.insert(0, self.points[0])
        #    self.control.insert(0, self.control[0])
        #    self.points.append(self.points[-1])
        #    self.control.append(self.control[-1])
    
    def evaluate(self, x):
        assert x >= self.control[0] and x <= self.control[-1]
        if x <= self.control[0]+1e-6: x = self.control[0]+1e-6
        if x >= self.control[-1]-1e-6: x = self.control[-1]-1e-6
        
        l = bisect.bisect_right(self.control, x) - 1
        if l == len(self.control)-1: l -= 1
        assert l >= 0
        assert self.control[l] <= x < self.control[l+1]
        
        n = self.degree
        d = [self.points[i] for i in xrange(l-n, l+1)]
        c = [self.control[i] for i in xrange(l-n+1, l-n+1+1+2*n-1)]
        lerp = lambda a, b, x: (1-x)*a + x*b
        for k in xrange(n):
            d = [lerp(d[i], d[i+1], (x - c[i+k])/(c[i+n] - c[i+k])) for i in xrange(n-k)]
        assert len(d) == 1
        return d[0]


import numpy
from matplotlib import pyplot
bs = CardinalBSpline([0, 0, 0, 6, 0, 0, 0], [-2, -2, -2, -2, -1, 0, 1, 2, 2, 2, 2], 3)
pyplot.plot(*zip(*[(x, bs.evaluate(x)) for x in numpy.linspace(-2, 2, 1000)]))
#pyplot.figure(2)
#for deg in [0, 1, 2]: #xrange(10):
#    bs = CardinalBSpline([1, 2, 3, 6, 5, 6, 7], numpy.linspace(0, 1, 7), deg)
#    pyplot.plot(*zip(*[(x, bs.evaluate(x)) for x in numpy.linspace(0, 1, 1000)]))
pyplot.show()
