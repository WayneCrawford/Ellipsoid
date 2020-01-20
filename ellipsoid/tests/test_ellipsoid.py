#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for testing the obspy.io.nordic functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import inspect
import os
import unittest
import numpy as np
from ellipsoid import Ellipsoid, Ellipse
from obspy.core.util.testing import ImageComparison
# import warnings
from itertools import cycle
import sys
sys.path.append('../')


class TestEllipsoidMethods(unittest.TestCase):
    """
    Test suite for ellipsoid operations.
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.testing_path = os.path.join(self.path, "data")

    def test_from_uncertainties(self):
        """
        Test Ellipsoid.from_uncertainties() (and from_covariance).
        """
        s_maj, s_min, s_int = 3, 1, 2
        self.assertEqual(Ellipsoid.from_uncertainties((1, 4, 3)),
                         Ellipsoid(4, 1, 3, -90, 0, 0))
        self.assertEqual(Ellipsoid.from_uncertainties((s_min, s_maj, s_int)),
                         Ellipsoid(s_maj, s_min, s_int, -90, 0, 0))
        self.assertEqual(Ellipsoid.from_uncertainties((s_min, s_int, s_maj)),
                         Ellipsoid(s_maj, s_min, s_int, -90, 90, 0))
        self.assertEqual(Ellipsoid.from_uncertainties((s_int, s_maj, s_min)),
                         Ellipsoid(s_maj, s_min, s_int, -90, 0, 90))
        self.assertEqual(Ellipsoid.from_uncertainties((s_int, s_min, s_maj)),
                         Ellipsoid(s_maj, s_min, s_int, 0, 90, 0))
        self.assertEqual(Ellipsoid.from_uncertainties((s_maj, s_min, s_int)),
                         Ellipsoid(s_maj, s_min, s_int, 0, 0, 0))
        self.assertEqual(Ellipsoid.from_uncertainties((s_maj, s_int, s_min)),
                         Ellipsoid(s_maj, s_min, s_int, 0, 0, 90))

    def test_ellipsoid_uncert_to_covariance(self):
        """
        Test Ellipsoid.from_uncert().
        """
        # Axes lengths [0 to 10]
        rng = np.random.default_rng()
        a, b, c = rng.integers(1, 10, size=3)
        # cross-correlation factors [0 to 1]
        ab = a * b * 0.5 * rng.uniform()
        ac = a * c * 0.5 * rng.uniform()
        bc = b * c * 0.5 * rng.uniform()
        self.assertEqual(Ellipsoid.from_covariance(
                            _cov_mat(a**2, b**2, c**2, ab, ac, bc)),
                         Ellipsoid.from_uncertainties((a, b, c), (ab, ac, bc)))
        self.assertEqual(Ellipsoid.from_covariance(
                            _cov_mat(a**2, c**2, b**2, ac, ab, bc)),
                         Ellipsoid.from_uncertainties((a, c, b), (ac, ab, bc)))
        self.assertEqual(Ellipsoid.from_covariance(
                            _cov_mat(b**2, a**2, c**2, ab, bc, ac)),
                         Ellipsoid.from_uncertainties((b, a, c), (ab, bc, ac)))
        self.assertEqual(Ellipsoid.from_covariance(
                            _cov_mat(b**2, c**2, a**2, bc, ab, ac)),
                         Ellipsoid.from_uncertainties((b, c, a), (bc, ab, ac)))
        self.assertEqual(Ellipsoid.from_covariance(
                            _cov_mat(c**2, a**2, b**2, ac, bc, ab)),
                         Ellipsoid.from_uncertainties((c, a, b), (ac, bc, ab)))
        self.assertEqual(Ellipsoid.from_covariance(
                            _cov_mat(c**2, b**2, a**2, bc, ac, ab)),
                         Ellipsoid.from_uncertainties((c, b, a), (bc, ac, ab)))

    def test_ellipsoid_crosscovariance(self):
        """
        Test Ellipsoids made with cross-covariances.
        """
        s_maj, s_min, s_int = 3, 1, 2
        factor = 0.999999
        # Just try one for now (not sure of my equations)
        self.assertEqual(
            Ellipsoid.from_uncertainties((s_maj, s_int, s_min),
                                         (s_maj * s_int * factor, 0, 0)),
            Ellipsoid(np.sqrt(s_maj**2 + s_int**2), 0.00235, s_min,
                      np.degrees(np.arctan2(s_int, s_maj)), 0, 0))

    def test_ellipsoid_fail(self):
        """
        Test deliberate fails.
        """
        # Put axes in the wrong order
        with self.assertRaises(AssertionError):
            Ellipsoid(2, 1, 3, 0, 0, 0)
        # make a too big cross-correlation
        with self.assertRaises(AssertionError):
            Ellipsoid.from_uncertainties((2, 1, 3), (3, 0, 0))
        with self.assertRaises(AssertionError):
            Ellipsoid.from_uncertainties((2, 1, 3), (0, 7, 0))
        with self.assertRaises(AssertionError):
            Ellipsoid.from_uncertainties((2, 1, 3), (0, 0, 4))

    def test_ellipsoid_to_from_covariance(self):
        """
        Test reciprocity of to_covariance() and from_covariance().
        """
        s_maj, s_min, s_int = 3, 1, 2
        rng = np.random.default_rng()
        # azi, plunge, rot = 0, 90 , 0
        for i in range(5):
            # azi, plunge, rot = rng.integers(1, 90, size=3)
            azi = rng.integers(-90, 360)
            plunge = rng.integers(-90, 360)
            rot = rng.integers(-90, 90)
            e = Ellipsoid(s_maj, s_min, s_int, azi, plunge, rot)
            # print(e)
            cov = e.to_covariance()
            # print(cov)
            self.assertEqual(e, Ellipsoid.from_covariance(cov))

    def test_ellipsoid_to_from_uncertainties(self):
        """
        Test reciprocity of to_uncerts() and from_uncertainties().
        """
        s_maj, s_min, s_int = 3, 1, 2
        rng = np.random.default_rng()
        azi, plunge, rot = rng.integers(1, 90, size=3)
        e = Ellipsoid(s_maj, s_min, s_int, azi, plunge, rot)
        uncerts, cross_covs = e.to_uncertainties()
        self.assertEqual(e, Ellipsoid.from_uncertainties(uncerts, cross_covs))

    def test_ellipsoid_to_ellipse(self):
        """
        Test Ellipsoid to Ellipse
        """
        s_maj, s_min, s_int = 5, 1, 3
        e = Ellipsoid(s_maj, s_min, s_int, 0, 0, 0)
        self.assertEqual(e.to_Ellipse(), Ellipse(s_maj, s_min, 0))
        e = Ellipsoid(s_maj, s_min, s_int, 90, 0, 0)
        self.assertEqual(e.to_Ellipse(), Ellipse(s_maj, s_min, 90))
        e = Ellipsoid(s_maj, s_min, s_int, 0, 90, 0)
        self.assertEqual(e.to_Ellipse(), Ellipse(s_int, s_min, 0))
        e = Ellipsoid(s_maj, s_min, s_int, 90, 90, 0)
        self.assertEqual(e.to_Ellipse(), Ellipse(s_int, s_min, 90))

    def test_ellipse_from_to_uncerts(self):
        """
        Verify ellipse is properly calculated and inverted from uncertainties

        tests Ellipse.from_uncerts and Ellipse.to_uncerts()
        """
        center = (20, 30)
        # First try simple cases without correlation
        x_errs = (0.5, 1.33, 1.0)
        y_errs = (1.33, 0.5, 1.0)
        for c_xy in [0, 0.2, 0.4, 0.6]:
            for (x_err, y_err) in zip(x_errs, y_errs):
                ell = Ellipse.from_uncertainties((x_err, y_err), c_xy, center)
                (x_err_out, y_err_out, c_xy_out, center_out) =\
                    ell.to_uncertainties()
                self.assertAlmostEqual(x_err, x_err_out)
                self.assertAlmostEqual(y_err, y_err_out)
                self.assertAlmostEqual(c_xy, c_xy_out)
                self.assertAlmostEqual(center, center_out)
        # Now a specific case with a finite covariance
        x_err = 0.5
        y_err = 1.1
        c_xy = -0.2149
        # Calculate ellipse
        ell = Ellipse.from_uncertainties((x_err, y_err), c_xy, center)
        self.assertAlmostEqual(ell.a, 1.120674193646)
        self.assertAlmostEqual(ell.b, 0.451762494786)
        self.assertAlmostEqual(ell.theta, 167.9407699)
        # Calculate covariance error from ellipse
        (x_err_out, y_err_out, c_xy_out, center_out) = ell.to_uncertainties()
        self.assertAlmostEqual(x_err, x_err_out)
        self.assertAlmostEqual(y_err, y_err_out)
        self.assertAlmostEqual(c_xy, c_xy_out)
        self.assertAlmostEqual(center, center_out)

    def test_ellipse_from_to_cov(self):
        """
        Verify ellipse is properly calculated and inverted using covariance

        tests Ellipse.from_uncerts and Ellipse.to_uncerts()
        """
        center = (20, 30)
        x_err = 0.5
        y_err = 1.1
        c_xy = -0.2149
        cov = [[x_err**2, c_xy], [c_xy, y_err**2]]
        # Calculate ellipse
        ell = Ellipse.from_covariance(cov, center)
        self.assertAlmostEqual(ell.a, 1.120674193646)
        self.assertAlmostEqual(ell.b, 0.451762494786)
        self.assertAlmostEqual(ell.theta, 167.9407699)
        # Calculate covariance error from ellipse
        cov_out, center_out = ell.to_covariance()
        self.assertAlmostEqual(cov[0][0], cov_out[0][0])
        self.assertAlmostEqual(cov[0][1], cov_out[0][1])
        self.assertAlmostEqual(cov[1][0], cov_out[1][0])
        self.assertAlmostEqual(cov[1][1], cov_out[1][1])

    def test_ellipse_from_uncertainties_baz(self, debug=False):
        """
        Verify alternative ellipse creator

        tests Ellipse.from_uncerts_baz
        """
        # Now a specific case with a finite covariance
        x_err = 0.5
        y_err = 1.1
        c_xy = -0.2149
        dist = 10
        baz = 90
        viewpoint = (5, 5)
        # Calculate ellipse
        ell = Ellipse.from_uncertainties_baz((x_err, y_err), c_xy,
                                             dist, baz, viewpoint)
        self.assertAlmostEqual(ell.a, 1.120674193646)
        self.assertAlmostEqual(ell.b, 0.451762494786)
        self.assertAlmostEqual(ell.theta, 167.9407699)
        self.assertAlmostEqual(ell.x, 15)
        self.assertAlmostEqual(ell.y, 5)
        baz = 180
        ell = Ellipse.from_uncertainties_baz(
            (x_err, y_err), c_xy, dist, baz, viewpoint)
        self.assertAlmostEqual(ell.x, 5)
        self.assertAlmostEqual(ell.y, -5)

    def test_ellipse_is_inside(self, debug=False):
        """
        Verify Ellipse.is_inside()
        """
        ell = Ellipse(20, 10, 90)
        self.assertIs(ell.is_inside((0, 0)), True)
        self.assertFalse(ell.is_inside((100, 100)))
        self.assertTrue(ell.is_inside((-19.9, 0)))
        self.assertTrue(ell.is_inside((19.9, 0)))
        self.assertFalse(ell.is_inside((-20.1, 0)))
        self.assertFalse(ell.is_inside((20.1, 0)))
        self.assertTrue(ell.is_inside((0, 9.9)))
        self.assertTrue(ell.is_inside((0, -9.9)))
        self.assertFalse(ell.is_inside((0, 10.1)))
        self.assertFalse(ell.is_inside((0, -10.1)))

    def test_ellipse_is_on(self, debug=False):
        """
        Verify Ellipse.is_on()
        """
        ell = Ellipse(20, 10, 90)
        self.assertFalse(ell.is_on((0, 0)))
        self.assertFalse(ell.is_on((100, 100)))
        self.assertTrue(ell.is_on((-20, 0)))
        self.assertTrue(ell.is_on((20, 0)))
        self.assertFalse(ell.is_on((-20.1, 0)))
        self.assertFalse(ell.is_on((20.1, 0)))
        self.assertTrue(ell.is_on((0, 10)))
        self.assertTrue(ell.is_on((0, -10)))
        self.assertFalse(ell.is_on((0, 10.1)))
        self.assertFalse(ell.is_on((0, -10.1)))

    def test_ellipse_subtended_angle(self, debug=False):
        """
        Verify Ellipse.subtended_angle()
        """
        ell = Ellipse(20, 10, 90)
        self.assertAlmostEqual(ell.subtended_angle((20, 0)), 180.)
        self.assertAlmostEqual(ell.subtended_angle((0, 0)), 360.)
        self.assertAlmostEqual(ell.subtended_angle((40, 0)), 32.204227503972)
        self.assertAlmostEqual(ell.subtended_angle((0, 40)), 54.623459848058)
        self.assertAlmostEqual(ell.subtended_angle((20, 10)), 89.9994270422)

    def test_ellipse_plot(self):
        """
        Compare a plot with one stored in the data/ directory
        """
        # Test single ellipse
        with ImageComparison(self.testing_path, 'plot_ellipse.png',
                             style='classic', reltol=10) as ic:
            Ellipse(20, 10, 90).plot(outfile=ic.name)

        # Test multi-ellipse figure
        with ImageComparison(self.testing_path, 'plot_ellipses.png',
                             style='classic', reltol=10) as ic:
            fig = Ellipse(20, 10, 90).plot(color='r')
            fig = Ellipse(20, 10, 45).plot(fig=fig, color='b')
            fig = Ellipse(20, 10, 0, center=(10, 10)).plot(fig=fig,
                                                           color='g')
            fig = Ellipse(20, 10, -45).plot(fig=fig, outfile=ic.name)

#     def test_ellipse_plot(self):
#         """
#         Test Ellipse.plot()
#
#         To generate test figures, used same commands after:
#         from ellipse import Ellipse
#         import matplotlib.pyplot as plt
#         plt.style.use('classic')
#         """
#         # Test single ellipse
#         with ImageComparison(self.testing_path, 'plot_ellipse.png',
#                              style='classic', reltol=10) as ic:
#             Ellipse(20, 10, 90).plot(outfile=ic.name)
#         # Test multi-ellipse figure
#         with ImageComparison(self.testing_path, 'plot_ellipses.png',
#                              style='classic', reltol=10) as ic:
#             fig = Ellipse(20, 10, 90).plot(color='r')
#             fig = Ellipse(20, 10, 45).plot(fig=fig, color='b')
#             fig = Ellipse(20, 10, 0, center=(10, 10)).plot(fig=fig,
#                           color='g')
#             fig = Ellipse(20, 10, -45).plot(fig=fig, outfile=ic.name)
#
    def test_ellipse_plot_tangents(self):
        """
        Test Ellipse.plot_tangents()
        """
        import matplotlib.pyplot as plt
        # Test single ellipse and point
        with ImageComparison(self.testing_path, 'plot_ellipse_tangents.png',
                             style='classic', reltol=10) as ic:
            Ellipse(20, 10, 90).plot_tangents((30, 30),
                                              color='b',
                                              print_angle=True,
                                              ellipse_name='Ellipse',
                                              outfile=ic.name)
        # Test multi-ellipse figure
        dist = 50
        fig = None
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = cycle(prop_cycle.by_key()['color'])
        step = 45
        with ImageComparison(self.testing_path, 'plot_ellipses_tangents.png',
                             style='classic', reltol=15) as ic:
            for angle in range(step, 360 + step - 1, step):
                x = dist * np.sin(np.radians(angle))
                y = dist * np.cos(np.radians(angle))
                ell = Ellipse(20, 10, 90, center=(x, y))
                if angle == 360:
                    outfile = ic.name
                else:
                    outfile = None
                fig = ell.plot_tangents((0, 0),
                                        fig=fig,
                                        color=next(colors),
                                        print_angle=True,
                                        ellipse_name='E{:d}'.format(angle),
                                        outfile=outfile)
        # Test multi-station figure
        fig = None
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = cycle(prop_cycle.by_key()['color'])
        with ImageComparison(self.testing_path,
                             'plot_ellipse_tangents_pts.png',
                             style='classic',
                             reltol=15) as ic:
            for angle in range(step, 360 + step - 1, step):
                x = dist * np.sin(np.radians(angle))
                y = dist * np.cos(np.radians(angle))
                ell = Ellipse(20, 10, 90)
                if angle == 360:
                    outfile = ic.name
                else:
                    outfile = None
                fig = ell.plot_tangents((x, y),
                                        fig=fig,
                                        color=next(colors),
                                        print_angle=True,
                                        pt_name='pt{:d}'.format(angle),
                                        outfile=outfile)


def _cov_mat(c_xx, c_yy, c_zz, c_xy, c_xz, c_yz):
    return np.array([[c_xx, c_xy, c_xz],
                     [c_xy, c_yy, c_yz],
                     [c_xz, c_yz, c_zz]])


def suite():
    return unittest.makeSuite(TestEllipsoidMethods, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
