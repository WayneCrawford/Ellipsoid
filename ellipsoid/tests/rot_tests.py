#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotation tests for ellipsoid
"""
from ellipsoid.ellipsoid import Ellipsoid
import numpy as np
import pprint

pp = pprint.PrettyPrinter()
np.set_printoptions(precision=2, suppress=True)
def main():
    step = 30
    for azi in np.arange(0, 90, step):
        for plunge in np.arange(0, 90, step):
            for rot in np.arange(0, 90, step):
                print_tofrom_cov(Ellipsoid(3, 1, 2, azi, plunge, rot))
#     print_tofrom_cov(Ellipsoid(3, 1, 2, 0, 0, 0))
#     print_tofrom_cov(Ellipsoid(3, 1, 2, 0, 0, 90))
#     print_tofrom_cov(Ellipsoid(3, 1, 2, 0, 90, 0))
#     print('The following gives same as 0,90,0, 180° flipped around z-axis')
#     print_tofrom_cov(Ellipsoid(3, 1, 2, 90, 90, 90))
#     print_tofrom_cov(Ellipsoid(3, 1, 2, 0, 90, 90))
#     print('The following gives same as 0,90,90, 180° flipped around z-axis')
#     print_tofrom_cov(Ellipsoid(3, 1, 2, 90, 90, 0))
#     
#     print_tofrom_cov(Ellipsoid(3, 1, 2, 90, 0, 0))
#     print_tofrom_cov(Ellipsoid(3, 1, 2, 90, 0, 90))
#     
#     rng = np.random.default_rng()
#     for i in range(10):
#         azi = rng.integers(-90, 90)
#         plunge = rng.integers(0, 90)
#         rot = rng.integers(0, 90)
#         print_tofrom_cov(Ellipsoid(3, 1, 2, azi, plunge, rot))

    
def print_tofrom_cov(el):
    debug = False
    # print(f'el =  {el}')
    cov = el.to_covariance()
    el2 = Ellipsoid.from_covariance(cov)
    print(f'{el}   {el2}')
    # print(f'el2 = {el2}')
    # print('')
    if debug:
        print('inp cov={}'.format(
            np.array2string(cov, separator=",", prefix=10*' ')))
        print('out cov={}'.format(
            np.array2string(el2.to_covariance(),
                            separator=",", prefix=10*' ')))
    
if __name__ == '__main__':
    main()
