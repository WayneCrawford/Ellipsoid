#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import random
from ellipsoid import Ellipsoid

title_fmt='{:2d}: {:50s}: '
i=0

i += 1
print('Using Ellipsoid.from_covariance()')
print('=================================')
title = 'A sphere of unit radius, should return Ellipse(1,1,1,x,x,x)'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_covariance(np.diag((1, 1, 1)))
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z = 1,4,3, non-rotated, should return Ellipse(4,1,3,0,90,0)'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_covariance(np.array([[1**2, 0, 0],
                                         [0, 4**2, 0],
                                         [0, 0, 3**2]]))
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z = 1,3,4, non-rotated, should return Ellipse(4,1,3,-90,-90,0)'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_covariance(np.array([[1**2, 0, 0],
                                         [0, 3**2, 0],
                                         [0, 0, 4**2]]))
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z = 3,1,4 dipping-nonrotated, should return Ellipse(4,1,3,-90,0,0)'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_covariance(np.array([[3**2, 0, 0],
                                         [0, 1**2, 0],
                                         [0, 0, 4**2]]))
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z = 3,4,1 rotated, should return Ellipse(4,1,3,0,90,90)'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_covariance(np.array([[3**2, 0, 0],
                                         [0, 4**2, 0],
                                         [0, 0, 1**2]]))
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z = 4,3,1 rotated, should return Ellipse(4,1,3,0,0,90)'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_covariance(np.array([[4**2, 0, 0],
                                         [0, 3**2, 0],
                                         [0, 0, 1**2]]))
print(el)
el.plot(title=title)


i += 1
print('\nUsing Ellipsoid.from_uncerts()')
print('=================================')
title = 'X,Y,Z =1,4,3, should return Ellipse(4,1,3,0,90,0)' 
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_uncerts([1., 4., 3.])
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z =1,4,3, max rotated in XY, should return Ellipse(4.1,0,3,0,x(0-90),0)'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_uncerts([1., 4., 3.], [3.999, 0, 0])
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z =1,4,3, max rotated all directions, should look like a straight line from all directions'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_uncerts([1., 4., 3.],
    0.999999*np.array([4.*1., 1.*3., 4.*3.]))
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z = 2,2,3, no rotation, should return Ellipse(3,2,2,-90,0,0)'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_uncerts([2., 2., 3.], [0, 0, 0])
print(el)
el.plot(title=title)

i += 1
# This one shows that the rotations aren't done right
title = 'X,Y,Z = 2,2,3, 45° (=atan(2/2)) XY rotation'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_uncerts([2., 2., 3.],[3.999, 0, 0])
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z = 2,2,3, 56° (=atan(3/2)) XZ rotation'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_uncerts([2., 2., 3.],[0, 5.999, 0])
print(el)
el.plot(title=title)

i += 1
title = 'X,Y,Z = 2,2,3, 56° (=atan(3/2)) YZ rotation,'
print(title_fmt.format(i, title), end='')
el = Ellipsoid.from_uncerts([2., 2., 3.],[0, 0, 5.999])
print(el)
el.plot(title=title)

print('Test Ellipse plotting')
print('=================================')

i += 1
title = "X,Y,Z = 1,3,2, Aligned "
print(title_fmt.format(i, title), end='')
el = Ellipsoid(3, 1, 2, 0, 90, 0)
print(el)
el.plot(title=title)

i += 1
title = "X,Y,Z = 2,3,1, aligned"
print(title_fmt.format(i, title), end='')
el = Ellipsoid(3, 1, 2, 0, 90, 90)
print(el)
el.plot(title=title)

i += 1
title = "X,Y,Z = 3,2,1, aligned"
print(title_fmt.format(i, title), end='')
el = Ellipsoid(3, 1, 2, 0, 0, -90)
print(el)
el.plot(title=title)

i += 1
title = "X,Y,Z = 2,1,3, aligned"
print(title_fmt.format(i, title), end='')
el = Ellipsoid(3, 1, 2, -90, 0, 0)
print(el)
el.plot(title=title)

i += 1
title = "X,Y,Z = 3,1,2, aligned"
print(title_fmt.format(i, title), end='')
el = Ellipsoid(3, 1, 2, 0, 0, 0)
print(el)
el.plot(title=title)

i += 1
title = "X,Y,Z = 1,2,3, aligned"
print(title_fmt.format(i, title), end='')
el = Ellipsoid(3, 1, 2, 90, 0, -90)
print(el)
el.plot(title=title)

i += 1
print('Test Ellipse conversion: to/from covariance')
print('=================================')
ax_len = random.randint(1,10,3)
ax_len_sorted = np.sort(ax_len)   # 3x1 biggest is last
ax_ang = random.randint(-90,90,3)   # 3x1, angles between -90 and 90.
el = Ellipsoid(ax_len_sorted[2], ax_len_sorted[0], ax_len_sorted[1], ax_ang[0], ax_ang[1], ax_ang[2])
print(el)
# 
print('to/from covariance')
print('=================================')

Ellipsoid.check_equi_covarinace(el)
# cov=el.to_cov()
# el2 = Ellipsoid.from_covariance(cov)
# assert el == el2  # Must write a routine to check equivalence
# print(el2)
# 
# print('to/from uncertainties')
# print('=================================')
# uncerts, xcovs = el.to_uncerts()
# el3 = Ellipsoid.from_uncerts(uncerts, xcovs)
# assert el == el3  # Must write a routine to check equivalence
# print(el3)


