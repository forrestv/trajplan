from __future__ import division

import bisect
import math
import time
import sys

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
        assert l-n >= 0
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

import random
random = random.Random('helloo')
spline_control_point_count = 20
spline_control_points = [numpy.array([
    math.cos(i/spline_control_point_count*math.pi),
    math.sin(i/spline_control_point_count*math.pi),
])+numpy.array([random.gauss(0, 1), random.gauss(0, 1)])*.04 for i in xrange(spline_control_point_count+1)]

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
in_range = lambda x, (lo, hi): lo <= x <= hi
slightly_enlarge_range = lambda (lo, hi): (lo-(hi-lo)*1e-6, hi+(hi-lo)*1e-6)
union_ranges = lambda (lo1, hi1), (lo2, hi2): (min(lo1, lo2), max(hi1, hi2))

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

N = 1001
ds = 1/(N-1)
ses = [i/(N-1) for i in xrange(N)]

def get_allowable_d2s_over_dt2_range(s_index, ds_over_dt):
    res = bs.evaluate(ses[s_index])
    
    # u = line_start + d2s/dt2 line_dir
    line_start = m * res['a'] * ds_over_dt**2 - w(res['p'], res['v'] * ds_over_dt)
    line_dir = m * res['v']
    
    allowed = (-inf, inf)
    for halfspace_normal, halfspace_dist in u_constraints:
        this_allowed = line_halfspace_intersection(line_start, line_dir, halfspace_normal, halfspace_dist)
        allowed = intersect_ranges(allowed, this_allowed)
    return allowed

def advance((s_index, ds_over_dt), d2s_over_dt2):
    # advance state over step of length ds with constant pseudoacceleration d2s_over_dt2
    # if pseudoacceleration would make our pseudospeed < 0, the pseudospeed is
    #     limited to exactly 0 and we return the pseudoacceleration
    #     that was needed to achieve that
    if ds_over_dt**2 + 2 * ds * d2s_over_dt2 <= 0:
        d2s_over_dt2 = -ds_over_dt**2 / 2 / ds
        ds_over_dt = 0
    else:
        ds_over_dt = math.sqrt(ds_over_dt**2 + 2 * ds * d2s_over_dt2)
    return (s_index + 1, ds_over_dt), d2s_over_dt2

def recede((s_index, ds_over_dt), d2s_over_dt2):
    # advance state over step of length ds with constant pseudoacceleration d2s_over_dt2
    # if pseudoacceleration would make our pseudospeed < 0, the pseudospeed is
    #     limited to exactly 0 and we return the pseudoacceleration
    #     that was needed to achieve that
    if ds_over_dt**2 + 2 * -ds * d2s_over_dt2 <= 0:
        d2s_over_dt2 = -ds_over_dt**2 / 2 / -ds
        ds_over_dt = 0
    else:
        ds_over_dt = math.sqrt(ds_over_dt**2 + 2 * -ds * d2s_over_dt2)
    return (s_index - 1, ds_over_dt), d2s_over_dt2

def get_d2s_over_dt2(ds_over_dt1, ds_over_dt2):
    return (ds_over_dt2**2 - ds_over_dt1**2) / (2 * ds)

def can_stop_from(x):
    if x is None: return False
    (s_index, ds_over_dt) = x
    assert s_index <= N-1
    while True:
        if ds_over_dt == 0: return True
        if s_index == N-1: return False
        rng = get_allowable_d2s_over_dt2_range(s_index, ds_over_dt)
        if not range_is_valid(rng): return False
        s_index, ds_over_dt = advance((s_index, ds_over_dt), rng[0])[0]

def advance_accelerating(state):
    if state is None: return None
    rng = get_allowable_d2s_over_dt2_range(*state)
    if not range_is_valid(rng): return None
    return advance(state, rng[1])[0]

def can_accelerate(state):
    return can_stop_from(advance_accelerating(state))

def multiadvance_accelerating(state, count):
    for i in xrange(count):
        state = advance_accelerating(state)
    return state

def multiadvance_decelerating(state, count):
    for i in xrange(count):
        rng = get_allowable_d2s_over_dt2_range(*state)
        assert range_is_valid(rng)
        state = advance(state, rng[0])[0]
    return state

def find_maximum_ds_over_dt(s_index):
    last_a = 0
    assert range_is_valid(get_allowable_d2s_over_dt2_range(s_index, last_a))
    a = 1
    while range_is_valid(get_allowable_d2s_over_dt2_range(s_index, a)):
        last_a = a
        a *= 2
    # a is invalid
    # breakpoint is in (last_a, a]; binary search to find it
    breakpoint_range = last_a, a
    for i in xrange(20):
        if range_is_valid(get_allowable_d2s_over_dt2_range(s_index, (breakpoint_range[0] + breakpoint_range[1])/2)):
            breakpoint_range = (breakpoint_range[0] + breakpoint_range[1])/2, breakpoint_range[1]
        else:
            breakpoint_range = breakpoint_range[0], (breakpoint_range[0] + breakpoint_range[1])/2
    return breakpoint_range[0]

if 0:
    pyplot.plot(*zip(*[(ses[s_index], find_maximum_ds_over_dt(s_index)) for s_index in xrange(N)]))
    pyplot.plot(*zip(*[(ses[s_index], .1*get_allowable_d2s_over_dt2_range(s_index, find_maximum_ds_over_dt(s_index))[0]) for s_index in xrange(N)]))
    pyplot.show()
    sys.exit()

start_time = time.time()

print 'forward'

ds_over_dt = 0
ds_over_dt_values = []
for s_index in xrange(N):
    rng = get_allowable_d2s_over_dt2_range(s_index, ds_over_dt)
    if not range_is_valid(rng):
        ds_over_dt = find_maximum_ds_over_dt(s_index)
        rng = get_allowable_d2s_over_dt2_range(s_index, ds_over_dt)
    ds_over_dt_values.append(ds_over_dt)
    (_, ds_over_dt), chosen = advance((s_index, ds_over_dt), rng[1])
    assert _ == s_index + 1
    assert chosen == rng[1]

p1 = ds_over_dt_values

print 'backward'

ds_over_dt = 0
ds_over_dt_values = []
for s_index in reversed(xrange(N)):
    rng = get_allowable_d2s_over_dt2_range(s_index, ds_over_dt)
    if not range_is_valid(rng):
        ds_over_dt = find_maximum_ds_over_dt(s_index)
        rng = get_allowable_d2s_over_dt2_range(s_index, ds_over_dt)
    ds_over_dt_values.append(ds_over_dt)
    (_, ds_over_dt), chosen = recede((s_index, ds_over_dt), rng[0])
    assert _ == s_index - 1
    assert chosen == rng[0]

p2 = ds_over_dt_values[::-1]
ds_over_dt_values = map(min, p1, p2)
d2s_over_dt2_values = map(get_d2s_over_dt2, ds_over_dt_values[:-1], ds_over_dt_values[1:])

if 0:
    for s_index, d2s_over_dt2 in enumerate(d2s_over_dt2_values):
        # d2s_over_dt2 applies over the s_index to s_index+1 interval
        left_rng = get_allowable_d2s_over_dt2_range(s_index, ds_over_dt_values[s_index])
        right_rng = get_allowable_d2s_over_dt2_range(s_index+1, ds_over_dt_values[s_index+1])
        assert in_range(d2s_over_dt2, slightly_enlarge_range(union_ranges(left_rng, right_rng)))

if 0:
    pyplot.plot(*zip(*[(ses[s_index], find_maximum_ds_over_dt(s_index)) for s_index in xrange(N)]))
    #pyplot.plot(ses, p1)
    #pyplot.plot(ses, p2)
    pyplot.plot(ses, ds_over_dt_values)
    pyplot.plot(numpy.array(ses[:-1]), d2s_over_dt2_values)
    pyplot.plot(*zip(*[(ses[s_index], min(get_allowable_d2s_over_dt2_range(s_index, ds_over_dt_values[s_index])[0], get_allowable_d2s_over_dt2_range(s_index+1, ds_over_dt_values[s_index+1])[0])) for s_index in xrange(N-1)]))
    pyplot.plot(*zip(*[(ses[s_index], max(get_allowable_d2s_over_dt2_range(s_index, ds_over_dt_values[s_index])[1], get_allowable_d2s_over_dt2_range(s_index+1, ds_over_dt_values[s_index+1])[1])) for s_index in xrange(N-1)]))
    pyplot.show()
    sys.exit()

print 'done'

t = 0
t_values = [t] # t_values[i] is defined as applying at ses[i]
for ds_over_dt1, ds_over_dt2 in zip(ds_over_dt_values[:-1], ds_over_dt_values[1:]):
    # circular integration..!
    # find new_t such that a circle goes through (s1, t) and (s2, new_t)
    #   and has slope 1/ds_over_dt1 at s1 and 1/ds_over_dt2 at s2.
    # avoids divide by zero that arises from standard integration.
    # consequences of inventing new math: unknown.
    t = t + ds * (1 - ds_over_dt1*ds_over_dt2 + math.sqrt((1+ds_over_dt1**2)*(1+ds_over_dt2**2)))/(ds_over_dt1+ds_over_dt2)
    t_values.append(t)

end_time = time.time()
print 'planning took', (end_time - start_time)/1e-3, 'ms'

result = []
for s_index in xrange(N):
    speval = bs.evaluate(ses[s_index])
    t = t_values[s_index]
    ds_over_dt = ds_over_dt_values[s_index]
    d2s_over_dt2 = d2s_over_dt2_values[s_index] if s_index < N-1 else 0
    p = speval['p']
    v = speval['v'] * ds_over_dt
    a = speval['a'] * ds_over_dt**2 + speval['v'] * d2s_over_dt2
    result.append(dict(
        s=ses[s_index],
        t=t,
        ds_over_dt=ds_over_dt,
        d2s_over_dt2=d2s_over_dt2, # distinct from the others (which are instantaneous), this quantity applies from this instant to the next
        p=p,
        v=v,
        a=a,
        u=m*a - w(p, v),
    ))

if 0:
    pyplot.plot(*zip(*[(x['t'], x['u'][0]) for x in result]))
    pyplot.plot(*zip(*[(x['t'], x['u'][1]) for x in result]))
    pyplot.show()
else:
    spline_sampled_points = [bs.evaluate(x)['p'] for x in numpy.linspace(0, 1, 100)]
    def animate(i):
        pyplot.cla()
        pyplot.gca().set_aspect('equal')
        pyplot.plot(*zip(*spline_sampled_points))
        pyplot.scatter(*zip(*spline_control_points))
        t = time.time() % result[-1]['t']
        i = min(xrange(N), key=lambda i: abs(result[i]['t'] - t))
        inst = result[i]
        pyplot.scatter(*zip(*[inst['p']]) + [80])
        #pyplot.arrow(*(list(inst['p']) + list(.1*inst['v'])))
        pyplot.arrow(*(list(inst['p']) + list(.1*inst['u'])))
        pyplot.plot(*zip(*[inst['p'] + .1*numpy.array(x) for x in [(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]]))

    fig = pyplot.figure()

    ani = animation.FuncAnimation(fig, animate, interval=1)

    pyplot.show()
