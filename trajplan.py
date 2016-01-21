from __future__ import division

import bisect

class CardinalBSpline(object):
    def __init__(self, points, control, degree):
        points = list(points)
        control = map(float, control)
        assert len(points) == len(control)
        self.points = points
        self.control = control
        self.degree = degree
        for i in xrange(self.degree):
            self.points.insert(0, self.points[0])
            self.control.insert(0, self.control[0])
            self.points.append(self.points[-1])
            self.control.append(self.control[-1])
    
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
        d_start = l-n
        lerp = lambda a, b, x: (1-x)*a + x*b
        for k in xrange(1, n+1):
            d = [lerp(d[i-1-d_start], d[i-d_start], (x-self.control[i])/(self.control[i+n+1-k] - self.control[i])) for i in xrange(l-n+k, l+1)]
            d_start = l-n+k
        assert len(d) == 1
        return d[l-d_start]



import numpy
from matplotlib import pyplot
for deg in [0, 1, 2]: #xrange(10):
    bs = CardinalBSpline([1, 2, 3, 6, 5, 6, 7], numpy.linspace(0, 1, 7), deg)
    pyplot.plot(*zip(*[(x, bs.evaluate(x)) for x in numpy.linspace(0, 1, 1000)]))
pyplot.show()
