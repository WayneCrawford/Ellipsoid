#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting tests to verify that ellipsoid and ellipse work correctly

Also used to generate plots used in test_ellipsoid.py
"""
import re
import os.path
# import numpy as np
from ellipsoid.ellipsoid import Ellipsoid
import matplotlib.pyplot as plt


def main():
    s_maj, s_min, s_int = 5, 1, 3
    for azi in range(0, 180, 30):
        plunge = 30
        rot = 30
        _plot3D(Ellipsoid(s_maj, s_min, s_int, azi, plunge, rot))
    for plunge in range(0, 180, 30):
        azi = 30
        rot = 30
        _plot3D(Ellipsoid(s_maj, s_min, s_int, azi, plunge, rot))
    for rot in range(0, 180, 30):
        azi = 30
        plunge = 30
        _plot3D(Ellipsoid(s_maj, s_min, s_int, azi, plunge, rot))
    # One value taken directly from Mayotte (greatest uncert is depth)
    _plot3D(Ellipsoid(2.68e+03, 1.27e+03, 2.12e+03, -91.3, 124, 104))


def slugify(value):
    """ Return string with non-ASCII characters replaced by '_' """
    value = re.sub(r'[^\w\s-]', '_', value.lower()).strip()
    return re.sub(r'[\s]+', '', value)


def _plot3D(el):
    """ Plot Ellipsoid and corresponding Ellipse in 3D """
    xy_el = el.to_Ellipse()
    title = el.__str__(as_ints=True) + ', ' + str(xy_el)
    print(title)
    fname = os.path.join('data',slugify(el.__str__(as_ints=True)) + '.png')
    fig = plt.figure()

    # View from Top
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    el.plot(fig=fig, viewpt=(180, 90), title='Top View')
    xy_el.plot(fig=fig)

    # 3D View from SSE
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    xy_el.plot(fig=fig)
    el.plot(fig=fig, viewpt=(150, 30), title='3D view')

    # View from South
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    el.plot(fig=fig, viewpt=(180, 0), title='South View')
    xy_el.plot(fig=fig)

    # View from East
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    el.plot(fig=fig, viewpt=(90, 0), title='East View')
    xy_el.plot(fig=fig)

    fig.suptitle(title)
    fig.show()
    fig.savefig(fname)


if __name__ == "__main__":
    main()
