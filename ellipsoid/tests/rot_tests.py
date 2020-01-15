#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotation tests for ellipsoid.Ellipsoid

I have run these for angles from -180 to 720, with rational and
irrational stepsizes between 2 and 30
"""
from ellipsoid.ellipsoid import Ellipsoid
import numpy as np

np.set_printoptions(precision=2, suppress=True)

def main():
    step = 3*np.pi
    step = 15
    azi_max = 180
    rot_max = 180
    itests, igood = 0, 0
    for azi in np.arange(0, azi_max, step):
        for plunge in np.arange(0, azi_max, step):
            for rot in np.arange(0, rot_max, step):
                el = Ellipsoid(3, 1, 2, azi, plunge, rot)
                itests += 1
                if print_tofrom_cov(el):
                    igood += 1
    print(f'{igood:d}/{itests:d} good tests')


def print_tofrom_cov(el):
    debug = False
    if debug:
        print('\n--- Test ---')
        print(f'el =  {el}')
    cov = el.to_covariance()
    el2 = Ellipsoid.from_covariance(cov)
    print(f'{el}   {el2} {el==el2}')
    # print(f'el2 = {el2}')
    # print('')
    if debug:
        print('inp cov={}'.format(
            np.array2string(cov, separator=",", prefix=10*' ')))
        print('out cov={}'.format(
            np.array2string(el2.to_covariance(),
                            separator=",", prefix=10*' ')))
    return el == el2


if __name__ == '__main__':
    main()
