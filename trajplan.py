from __future__ import division

class CardinalBSpline(object):
    def __init__(self, points, degree):
        self.points = points
        self.degree = degree
    
    def evaluate(self, x):
        l = int(x)
        n = self.degree
        if l-n < 0: return None
        d = [self.points[i] for i in xrange(l-n, l+1)]
        d_start = l-n
        lerp = lambda a, b, x: (1-x)*a + x*b
        for k in xrange(1, n+1):
            d = [lerp(d[i-1-d_start], d[i-d_start], (x-i)/(n+1-k)) for i in xrange(l-n+k, l+1)]
            d_start = l-n+k
        return d[l-d_start]



import numpy
from matplotlib import pyplot
for deg in xrange(5):
    bs = CardinalBSpline([1, 2, 3, 6, 5, 6, 7], deg)
    pyplot.plot(*zip(*[(x, bs.evaluate(x)) for x in numpy.linspace(0, 7, 1000)[:-1]]))
pyplot.show()
