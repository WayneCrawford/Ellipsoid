#! /bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation as R
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
        
        if plunge, azimuth, rotation == 0, 0, 0:
            semi-major is along y (S-N)
            semi-minor is along x (W-E)
            
        Note that the quakeml specification says that x is the S-N
        direction and y the W-E direction, but their figure 4 shows
        a geometry in which y would correspond to S-N and x to E-W.
        They do not say whether the semi-minor axis would correspond
        to z or to E-W if all angles==0
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
        #print(evecs)

        assert np.all(evals > 0), 'Covariance matrix is not positive definite'
        #assert np.allclose(np.linalg.norm(evecs),[1.,1.,1.]), 'Eigenvectors are not unit length'

        if debug:
            print(evecs)
            print(evals)

        #Semi-major axis lengths
        s_min, s_inter, s_maj = np.sqrt(evals)
        #print(s_min,s_inter,s_maj)

        
        # Calculate angles of semi-major axis
        # From wikipedia (z-x'-y'' convention, left-hand rule)
        Y1, Y2, Y3 = evecs[:, 0]  # Unit semi-minor axis ("Y")
        X1, X2, X3 = evecs[:, 2]  # Unit semi-major axis ("X")
        #print(X1,X2,X3,Y1,Y2,Y3)
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

        Internal because x, y and z are in ConfidenceEllipsoid order
        
        ARE YOU SURE?  I THOUGHT THEY WERE y, x, z ORDER?"""
        eigvals = (self.semi_major_axis_length,
                   self.semi_minor_axis_length,
                   self.semi_intermediate_axis_length)
        # https://math.stackexchange.com/questions/1403126/what-is-the-general-equation-equation-for-rotated-ellipsoid
        RZ_azi = Ellipsoid.__ROT_RH_azi(np.radians(self.major_axis_azimuth))
        RY_plunge = Ellipsoid.__ROT_RH_plunge(np.radians(self.major_axis_plunge))
        RX_rot = Ellipsoid.__ROT_RH_rot(np.radians(self.major_axis_rotation))

        #print(RZ_azi, RY_plunge, RX_rot)

        eigvecs = RZ_azi * RY_plunge * RX_rot
        # print(eigvecs)
        return eigvals, eigvecs


    def __rotmat(self):
        """
        Return rotation matrix of ellipsoid
        """
        r = R.from_euler('z', self.major_axis_azimuth, degrees=True) *\
            R.from_euler('x', self.major_axis_plunge, degrees=True) *\
            R.from_euler('y', self.major_axis_rotation, degrees=True)
        return r

    @staticmethod
    def __ROT_RH_azi(azi):
        """Right handed rotation matrix for "azimuth" in RADIANS"""
        c_azi, s_azi = np.cos(azi), np.sin(azi)
        return np.array([[1, 0, 0],
                       [0, c_azi, -s_azi],
                       [0, s_azi, c_azi]])
    @staticmethod
    def __ROT_RH_plunge(plunge):
        """Right handed rotation matrix for "plunge" in RADIANS"""
        c_plunge, s_plunge = np.cos(plunge), np.sin(plunge)      
        return np.array([[c_plunge, 0, s_plunge],
                       [0, 1, 0],
                       [-s_plunge, 0, c_plunge]])
    @staticmethod
    def __ROT_RH_rot(rot):
        """Right handed rotation matrix for "rotation" in RADIANS"""
        c_rot, s_rot = np.cos(rot), np.sin(rot)
        return np.array([[c_rot, -s_rot, 0],
                       [s_rot, c_rot, 0],
                       [0, 0, 1]])

    def to_covariance(self, debug=False):
        """Return covariance matrix corresponding to ellipsoid

        Uses eigenvals*cov=eigenvecs*cov
        """
        eigvals, eigvecs = self.__to_eigen()
        cov = eigvecs * np.diag(eigvals) * np.linalg.inv(eigvecs)

        # THIS COMMENT IS WRITTEN FOR 2D!!!
        # Convert to standard coordinates (x=E, y=N). Transform covariance
        # matrix to CovarianceEllipse coordinates (x=N,y=E)
        #temp = cov(0, 1)   # old c_xz, becomes c_yz
        #cov[0, 2], cov[2, 0] = cov[1, 2], cov[1, 2]
        #cov[1, 2], cov[2, 1] = temp, temp
        #temp = cov[0, 0]  # old c_xx, becomes c_yy
        #cov[0, 0] = cov[1, 1]
        #cov[1, 1] = temp

        return cov

    def to_XYEllipse(self, debug=False):
        """
        Return XY-ellipse corresponding to Ellipsoid
        """
        a,b = self.semi_major_axis_length, self.semi_minor_axis_length
        theta = 0
        center = (0,0)
        return a,b,theta,center


    #def to_XYEllipse(self, debug=False):
        """Return XY-plane Ellipse corresponding to Ellipsoid

        Should probably make a generic to_Ellipse, allowing one 
        to extract the Ellipse viewed from any angle, then to_XYEllipse
        would call this code from the appropriate view angle
        """
    #    print('to_XYEllipse is not yet written!')

    def to_xyz(self, debug=False):
        """Return xyz and covariances corresponding to ellipsoid """
        cov = self.to_covariance()
        errors = np.sqrt(np.diag(cov))
        cross_covs = cov[0, 1], cov[0, 2], cov[1, 2]
        return errors, cross_covs

    def check_equi_covarinace(self, debug=False):
        """
        check equaivalne between from/to covariance
        """
        cov = self.to_covariance()
        print(cov)
        ell = self.from_covariance(cov)
        print(ell)
        #assert np.all(ell() == self.()), 'not equal' 

    def plot_old(self, title=None, debug=False):
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
        cov1 = self.to_covariance()
        eigvals, eigvecs = np.linalg.eig(cov1)

        # Width, height and depth of ellipsoid
        #rx, ry, rz = (self.semi_major_axis_length,
        #           self.semi_minor_axis_length,
        #           self.semi_intermediate_axis_length)

        rx,ry,rz = eigvals
        #print(rx,ry,rz)


        # Get the xyz points for plotting
        # Cartesian coordinates that correspond to the spherical angles:
        X = rx * np.outer(np.cos(theta), np.sin(phi))
        Y = ry * np.outer(np.sin(theta), np.sin(phi))
        Z = rz * np.outer(np.ones(np.size(theta)), np.cos(phi))

        # Rotate ellipsoid
        old_shape = X.shape
        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
        #print(X.shape, Y.shape, Z.shape)
        X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
        #print(X.shape, Y.shape, Z.shape)

        # Add in offsets, flipping X and Y to correspond to "external"
        # convention
        #X = Y + self.center[0]
        #Y = X + self.center[1]
        #Z = Z + self.center[2]

        # Plot
        ax.plot_wireframe(X, Y, Z, alpha=0.3, color='r')
        plt.xlabel('x(E)')
        plt.ylabel('y(N)')
        if title:
            plt.title(title)
        _set_axes_equal(ax)
        plt.show()


    def plot(self, title=None, debug=False):
        """
        Plots ellipsoid

        https://stackoverflow.com/questions/7819498/plotting-ellipsoid-
              with-matplotlib
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Make set of spherical angles to draw our ellipsoid
        n_points = 100
        theta = np.linspace(0, 2*np.pi, n_points)
        phi = np.linspace(0, np.pi, n_points)

        # Get the xyz points for plotting
        # Cartesian coordinates that correspond to the spherical angles:
        X = self.semi_minor_axis_length * np.outer(np.cos(theta), np.sin(phi))
        Y = self.semi_major_axis_length * np.outer(np.sin(theta), np.sin(phi))
        Z = self.semi_intermediate_axis_length *\
            np.outer(np.ones(np.size(theta)), np.cos(phi))

        old_shape = X.shape
        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        r = self.__rotmat()
        # Rotate ellipsoid
        XYZ_rot = r.apply(np.array([X,Y,Z]).T)
        X_rot, Y_rot, Z_rot = XYZ_rot[:,0].reshape(old_shape),\
                              XYZ_rot[:,1].reshape(old_shape),\
                              XYZ_rot[:,2].reshape(old_shape)
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X_rot, Y_rot, Z_rot, alpha=0.3, color='r')
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
    limits = np.array([ax.get_xlim3d(),
                       ax.get_ylim3d(),
                       ax.get_zlim3d()])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

