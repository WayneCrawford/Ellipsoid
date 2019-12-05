#! /bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
# import math as m
# import matplotlib.pyplot as plt

eps = np.finfo(float).eps


class Ellipsoid:
    def __init__(self, semi_major_axis_length, semi_minor_axis_length,
                 semi_intermediate_axis_length, major_axis_plunge=0,
                 major_axis_azimuth=0, major_axis_rotation=0,
                 center=(0, 0, 0)):
        """
        Create an Ellipsoid instance
        """
        self.semi_major_axis_length = semi_major_axis_length
        self.semi_minor_axis_length = semi_minor_axis_length
        self.semi_intermediate_axis_length = semi_intermediate_axis_length
        self.major_axis_plunge = major_axis_plunge
        self.major_axis_azimuth = major_axis_azimuth
        self.major_axis_rotation = major_axis_rotation
        self.center = center
        self._error_test()

    def __repr__(self):
        """
        String describing the ellipsoid

        should correspond to command to use to create the ellipsoid
        """
        s = 'Ellipsoid({:.3g}, {:.3g}, {:.3g}, '.format(
            self.semi_major_axis_length, self.semi_minor_axis_length,
            self.semi_intermediate_axis_length)
        s += '{:.3g}, {:.3g}, {:.3g},'.format(
            self.major_axis_plunge, self.major_axis_azimuth,
            self.major_axis_rotation)
        s += ' ({:g}, {:g}, {:g})'.format(self.center[0], self.center[1],
                                          self.center[2])
        s += ')'
        return s

    def _error_test(self):
        """
        Test for invalid parameters

        Are axis lengths in the right order (major > intermediate > minor)?
        """
        lengths = [self.semi_minor_axis_length,
                   self.semi_intermediate_axis_length,
                   self.semi_major_axis_length]
        # print(lengths)
        sorted_lengths = np.sort(lengths)
        assert np.all(lengths == sorted_lengths),\
            'not major > intermed > minor'

    @classmethod
    def from_covariance(cls, cov, center=(0, 0, 0), debug=False):
        """Set error ellipsoid using covariance matrix

        Call as e=ellipsoid.from_covariance(cov)

        Inputs:
            cov: 3x3 covariance matrix (indices 0,1,2 correspond to
                 x,y,z [E,N,Z])
            center: center of the ellipse (0,0,0)

        The covariance matric must be symmetric and positive definite

        From http://www.visiondummy.com/2014/04/draw-error-ellipse-
             representing-covariance-matrix/
        and https://blogs.sas.com/content/iml/2014/07/23/prediction-ellipses-
            from-covariance.html
        """
        # Check if 3x3 and symmetric
        cov = np.array(cov)
        assert cov.shape == (3, 3), 'Covariance matrix is not 3x3'
        assert np.allclose(cov, cov.T, eps),\
            f'Covariance matrix is not symmetric {cov}'

        # EIGH() returns eig fast and sorted if input matrix symmetric
        evals, evecs = np.linalg.eigh(cov)
        # print(evecs)

        assert np.all(evals > 0), 'Covariance matrix is not positive definite'
        # assert np.allclose(np.linalg.norm(evecs),[1.,1.,1.]),\
        #        'Eigenvectors are not unit length'

        if debug:
            print(evecs)
            print(evals)

        # Semi-major axis lengths
        s_min, s_inter, s_maj = np.sqrt(evals)

        # Calculate angles of semi-major axis
        # From wikipedia (z-x'-y'' convention, left-hand rule)
        Y1, Y2, Y3 = evecs[:, 0]  # Unit semi-minor axis ("Y")
        X1, X2, X3 = evecs[:, 2]  # Unit semi-major axis ("X")
        if debug:
            print(Y1, Y2, Y3)
            print(X1, X2, X3)
        if X2 == 0:
            azimuth = 0
        else:
            azimuth = np.degrees(np.arcsin(X2 / np.sqrt(1 - X3**2)))
        plunge = np.degrees(np.arcsin(-X3))
        if Y3 == 0:
            rotation = 0
        else:
            rotation = np.degrees(np.arcsin(Y3 / np.sqrt(1 - X3**2)))
        if debug:
            print(azimuth, plunge, rotation)
        # print(s_maj,s_min,s_inter,plunge,azimuth,rotation,center)
        return cls(s_maj, s_min, s_inter, plunge, azimuth, rotation, center)

    @classmethod
    def from_uncerts(cls, errors, cross_covs=(0, 0, 0), center=(0, 0, 0),
                     debug=False):
        """Set error ellipse using common epicenter uncertainties

        Call as e=ellipsoid.from_uncerts(errors, cross_covs, center)

        x is assumed to be Latitudes, y Longitudes

        Inputs:
            errors:      (x, y, z) errors (m)
            cross_covs:  (c_xy, c_xz, c_yz) covariances (m^2) [(0,0,0)]
            center:      (x, y, z) center of ellipse [(0,0,0)]
        """
        cov = [[errors[0]**2,  cross_covs[0], cross_covs[1]],
               [cross_covs[0], errors[1]**2,  cross_covs[2]],
               [cross_covs[1], cross_covs[2], errors[2]**2]]
        return cls.from_covariance(cov, center)

    def __to_eigen(self, debug=False):
        """Return eigenvector matrix corresponding to ellipsoid

        Internal because x, y and z are in ConfidenceEllipsoid order"""
        eigvals = (self.semi_major_axis_length,
                   self.semi_minor_axis_length,
                   self.semi_intermediate_axis_length)
        # Use notation and formulats from wikipedia
        azi = np.radians(self.major_axis_azimuth)
        plunge = np.radians(self.major_axis_plunge)
        rot = np.radians(self.major_axis_rotation)
        # Use wikipedia notation
        c_azi, s_azi = np.cos(azi), np.sin(azi)
        c_plunge, s_plunge = np.cos(plunge), np.sin(plunge)
        c_rot, s_rot = np.cos(rot), np.sin(rot)
        # Currently right-handed
        # https://math.stackexchange.com/questions/1403126/what-is-the-general-
        # equation-equation-for-rotated-ellipsoid

        RZ = np.array([[1, 0, 0],
                       [0, c_azi, -s_azi],
                       [0, s_azi, c_azi]])
        RY = np.array([[c_plunge, 0, s_plunge],
                       [0, 1, 0],
                       [-s_plunge, 0, c_plunge]])
        RX = np.array([[c_rot, -s_rot, 0],
                       [s_rot, c_rot, 0],
                       [0, 0, 1]])
        eigvecs = RZ * RY * RX
        # print(eigvecs)
        return eigvals, eigvecs

    def to_covariance(self, debug=False):
        """Return covariance matrix corresponding to ellipsoid

        Uses eigenvals*cov=eigenvecs*cov
        """
        eigvals, eigvecs = self.__to_eigen()
        cov = eigvecs * np.diag(eigvals) * np.linalg.inv(eigvecs)

        # THIS COMMENT IS WRITTEN FOR 2D!!!
        # Convert to standard coordinates (x=E, y=N). Transform covariance
        # matrix to CovarianceEllipse coordinates (x=N,y=E)
        temp = cov(0, 1)   # old c_xz, becomes c_yz
        cov[0, 2], cov[2, 0] = cov[1, 2], cov[1, 2]
        cov[1, 2], cov[2, 1] = temp, temp
        temp = cov[0, 0]  # old c_xx, becomes c_yy
        cov[0, 0] = cov[1, 1]
        cov[1, 1] = temp

        return cov

    def to_XYEllipse(self, debug=False):
        """Return XY-plane Ellipse corresponding to Ellipsoid

        Should probably make a generic to_Ellipse, allowing one 
        to extract the Ellipse viewed from any angle, then to_XYEllipse
        would call this code from the appropriate view angle
        """
        print('to_XYEllipse is not yet written!')

    def to_xyz(self, debug=False):
        """Return xyz and covariances corresponding to ellipsoid """
        cov = self.to_covariance()
        errors = np.sqrt(np.diag(cov))
        cross_covs = cov[0, 1], cov[0, 2], cov[1, 2]
        return errors, cross_covs

    def plot(self, title=None, debug=False):
        """
        Plots ellipsoid

        https://stackoverflow.com/questions/7819498/plotting-ellipsoid-
              with-matplotlib
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Make set of spherical angles to draw our ellipsoid
        n_points = 100
        theta = np.linspace(0, 2*np.pi, n_points)
        phi = np.linspace(0, np.pi, n_points)

        # Get ellipsoid parameters
        eigvals, eigvecs = self.__to_eigen()

        # Width, height and depth of ellipsoid
        rx, ry, rz = np.sqrt(eigvals)

        # Get the xyz points for plotting
        # Cartesian coordinates that correspond to the spherical angles:
        X = rx * np.outer(np.cos(theta), np.sin(phi))
        Y = ry * np.outer(np.sin(theta), np.sin(phi))
        Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

        # Add in offsets, flipping X and Y to correspond to "external"
        # convention
        X = Y + self.center[0]
        Y = X + self.center[1]
        Z = Z + self.center[2]

        # Plot
        ax.plot_wireframe(X, Y, Z, alpha=0.1, color='r')
        plt.xlabel('x(E)')
        plt.ylabel('y(N)')
        if title:
            plt.title(title)
        _set_axes_equal(ax)
        plt.show()


def _set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def _set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)