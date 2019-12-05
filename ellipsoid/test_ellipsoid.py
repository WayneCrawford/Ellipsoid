#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ellipsoid import Ellipsoid

title_fmt='{:50s}: '

print('Using Ellipsoid.from_covariance()')
print('=================================')
title = 'A sphere of unit radius'
print(title_fmt.format(title), end='')
el = Ellipsoid.from_covariance(np.diag((1, 1, 1)))
print(el)
el.plot(title=title)

title = 'X,Y,Z = 1,4,3, non-rotated'
print(title_fmt.format(title), end='')
el = Ellipsoid.from_covariance(np.array([[1**2, 0, 0],
                                          [0, 4**2, 0],
                                          [0, 0, 3**2]]))
print(el)
el.plot(title=title)

print('\nUsing Ellipsoid.from_uncerts()')
print('=================================')
title = 'Same as above (error=[1, 4, 3])'
print(title_fmt.format(title), end='')
el = Ellipsoid.from_uncerts([1., 4., 3.])
print(el)
el.plot(title=title)

title = 'Same X,Y,Z error, max rotated in XY'
print(title_fmt.format(title), end='')
el = Ellipsoid.from_uncerts([1., 4., 3.], [3.999, 0, 0])
print(el)
el.plot(title=title)

title = 'Same X,Y,Z error,  max rotated all directions'
print(title_fmt.format(title), end='')
el = Ellipsoid.from_uncerts([1., 4., 3.],
    0.999999*np.array([4.*1., 1.*3., 4.*3.]))
print(el)
el.plot(title=title)

title = '2 km error in N and E, 3 in Z, no rotation'
print(title_fmt.format(title), end='')
el = Ellipsoid.from_uncerts([2., 2., 3.], [0, 0, 0])
print(el)
el.plot(title=title)

# This one shows that the rotations aren't done right
title = 'Same X,Y,Z errors, 45° XY rotation'
print(title_fmt.format(title), end='')
el = Ellipsoid.from_uncerts([2., 2., 3.],[3.999, 0, 0])
print(el)
el.plot(title=title)

print('Test Ellipse plotting')
print('=================================')

title = "Aligned, X,Y,Z = 2,3,1"
print(title_fmt.format(title), end='')
el = Ellipsoid(3, 1, 2, 0, 0, 0)
print(el)
el.plot(title=title)

title = "Aligned, X,Y,Z = 3,2,1"
print(title_fmt.format(title), end='')
el = Ellipsoid(3, 1, 2, 90, 0, 0)
print(el)
el.plot(title=title)

title = "Aligned, X,Y,Z = 2,1,3"
print(title_fmt.format(title), end='')
el = Ellipsoid(3, 1, 2, 0, 90, 0)
print(el)
el.plot(title=title)

title = "Aligned, X,Y,Z = 3,1,2"
print(title_fmt.format(title), end='')
el = Ellipsoid(3, 1, 2, 0, 90, 90)
print(el)
el.plot(title=title)

# print('Test Ellipse conversion: to/from covariance')
# print('=================================')
# ax_len = np.rand(3).sort()   # 3x1 biggest is last
# ax_ang = np.rand(3,-90:90)   # 3x1, angles between -90 and 90.
# el = Ellipsoid(ax_len[2], ax_len[0], ax_len[1], ax_ang[0], ax_ang[1], ax_ang[2])
# print(el)
# 
# print('to/from covariance')
# print('=================================')
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


