#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting tests to verify that ellipsoid and ellipse work correctly

Also used to generate plots used in test_ellipsoid.py
"""
import re
# import numpy as np
from ellipsoid.ellipsoid import Ellipsoid
import matplotlib.pyplot as plt


def main():
    _plot3D(Ellipsoid(3, 1, 2, -45, 45, 0))

    # N,E,Z = 3,1,2, aligned"
    _plot3D(Ellipsoid(3, 1, 2, 0, 0, 0))

    # N,E,Z = 3,2,1, aligned"
    _plot3D(Ellipsoid(3, 1, 2, 0, 0, 90))

    # N,E,Z = 2,1,3, aligned"
    _plot3D(Ellipsoid(3, 1, 2, 0, 90, 0))

    # N,E,Z = 1,2,3, aligned"
    _plot3D(Ellipsoid(3, 1, 2, 0, 90, -90))

     # Non-aligned
    _plot3D(Ellipsoid(3, 1, 2, 30, 30, 0))

    _plot3D(Ellipsoid(3, 1, 2, 30, 60, 0))
   
    _plot3D(Ellipsoid(3, 1, 2, 45, 45, 0))
    

    _plot3D(Ellipsoid(3, 1, 2, 60, 60, 0))
    
    _plot3D(Ellipsoid(3, 1, 2, 60, 30, 0))
    
   # N,E,Z = 1,3,2, Aligned
    _plot3D(Ellipsoid(3, 1, 2, 90, 0, 0))

    # N,E,Z = 2,3,1, aligned
    _plot3D(Ellipsoid(3, 1, 2, 90, 0, 90))


def slugify(value):
    """ Return string with non-ASCII characters replaced by '_' """
    value = re.sub(r'[^\w\s-]', '_', value.lower()).strip()
    return re.sub(r'[\s]+', '', value)


def _plot3D(el):
    """ Plot Ellipsoid and corresponding Ellipse in 3D """
    xy_el = el.to_Ellipse()
    title = el.__str__(as_ints=True) + ', ' + str(xy_el)
    print(title)
    fname = slugify(el.__str__(as_ints=True)) + '.png'
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
