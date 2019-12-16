#! /bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

eps = np.finfo(float).eps


class Ellipsoid:
    def __init__(self, semi_major_axis_length, semi_minor_axis_length,
                 semi_intermediate_axis_length, major_axis_azimuth=0,
                 major_axis_plunge=0, major_axis_rotation=0,
                 center=(0, 0, 0)):
        """
        Create an Ellipsoid instance using QuakeML parameterization

        The QuakeML manual states
        'The three Tait-Bryan rotations are performed as follows:
            (i) a rotation about the Z axis with angle ψ (heading, or azimuth);
            (ii) a rotation about the Y axis with angle φ (elevation, or
                 plunge);
            (iii) a rotation about the X axis with angle θ (bank).
            Note that in the case of Tait-Bryan angles, the rotations are
            performed about the ellipsoid’s axes, not about the axes of the
            fixed (x, y, z) Cartesian system.... Note that [the x-y geometry]
            can be interpreted as a hypothetical view from the interior of the
            Earth to the inner face of a shell representing Earth’s surface'

        if plunge, azimuth, rotation == 0, 0, 0:
            semi-major is along x (S-N)
            semi-minor is along y (W-E) [QuakeML document does not specify]
        """
        self.semi_major_axis_length = semi_major_axis_length
        self.semi_minor_axis_length = semi_minor_axis_length
        self.semi_intermediate_axis_length = semi_intermediate_axis_length
        self.major_axis_azimuth = major_axis_azimuth
        self.major_axis_plunge = major_axis_plunge
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
            self.major_axis_azimuth, self.major_axis_plunge,
            self.major_axis_rotation)
        s += ' ({:g}, {:g}, {:g})'.format(self.center[0], self.center[1],
                                          self.center[2])
        s += ')'
        return s

    def __eq__(self, other):
        """
        Returns true if two Ellipsoids are equal
        """
        if not abs((self.semi_major_axis_length - other.semi_major_axis_length)
                   / self.semi_major_axis_length) < 1e-5:
            return False
        if not abs((self.semi_minor_axis_length - other.semi_minor_axis_length)
                   / self.semi_minor_axis_length) < 1e-2 :
            return False
        if not abs((self.semi_intermediate_axis_length -
                    other.semi_intermediate_axis_length) /
                        abs(self.semi_intermediate_axis_length)) < 1e-2:
            return False
        if not self.center == other.center:
            return False
        if not np.equal(self.__rotmat().as_rotvec().round(5),
                        other.__rotmat().as_rotvec().round(5)).all():
            return False
        return True

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
                 x,y,z.  For geographic data, x=N, y=E,and Z=depth (to view
                 from above, use an elevation < 0))
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
        assert np.all(evals > 0),\
            f'Covariance matrix {cov} is not positive definite'
        # assert np.allclose(np.linalg.norm(evecs),[1.,1.,1.]),
        #                    'Eigenvectors are not unit length'

        if debug:
            print()
            print(evecs)
            print(evals)

        # print(s_min,s_inter,s_maj)

        # Calculate angles of semi-major axis
        # From wikipedia (z-x'-y'' convention, left-hand rule)
        Y1, Y2, Y3 = evecs[:, 0]  # Unit semi-minor axis ("Y")
        X1, X2, X3 = cls._choose_semi_major(cov, evecs, evals)
        if debug:
            print(Y1, Y2, Y3)
            print(X1, X2, X3)
        azimuth, plunge, rotation = cls._calc_rotation_angles(X2, X3, Y3)

        # Semi-major axis lengths
        s_min, s_inter, s_maj = np.sqrt(evals)
        return cls(s_maj, s_min, s_inter, azimuth, plunge, rotation, center)

    @classmethod
    def from_uncerts(cls, errors, cross_covs=(0, 0, 0), center=(0, 0, 0),
                     debug=False):
        """Set error ellipse using common epicenter uncertainties

        Call as e=ellipsoid.from_uncerts(errors, cross_covs, center)

        x=N, y=E, z=Depth

        Inputs:
            errors:      (x, y, z) errors (m)
            cross_covs:  (c_xy, c_xz, c_yz) covariances (m^2) [(0,0,0)]
            center:      (x, y, z) center of ellipse [(0,0,0)]
        """
        cov = [[errors[0]**2,  cross_covs[0], cross_covs[1]],
               [cross_covs[0], errors[1]**2,  cross_covs[2]],
               [cross_covs[1], cross_covs[2], errors[2]**2]]
        return cls.from_covariance(cov, center)

    @staticmethod
    def _choose_semi_major(cov, evecs, evals):
        """
        Calculate semi-major axis unit vector

        :param cov: covariance matrix
        :param evals: eigenvalues sorted from smallest to largest [1x3]
        :param evecs: corresponding eigenvectors (1/row)
        """
        # WCC: Why all these specific cases resulting in taking evecs[:,1]
        # (intermediate axis) instead of just taking evecs[:,2]?
        if not cov[0, 1] == 0 and evals[2] <= cov[2, 2]:     # XY rotation
            return evecs[:, 1]
        elif not cov[0, 2] == 0 and evals[2] <= cov[1, 1]:   # XZ rotation
            return evecs[:, 1]
        elif not cov[1, 2] == 0 and evals[2] <= cov[0, 0]:  # YZ rotation
            return evecs[:, 1]
        return evecs[:, 2]

    @staticmethod
    def _calc_rotation_angles(X2, X3, Y3, debug=False):
        """
        Calculate rotation angles from semi-major & -minor components
        
        WCC: PLEASE EXPLAIN HOW THIS WORKS.  CAN IT BE DONE MORE SIMPLY USING
        scipy.spatial.transform.Rotation?
        WCC: WHY IS PLUNGE ALWAYS NEGATIVE?
        """
        if X2 == 0:
            azimuth = 0
        else:
            azimuth = np.degrees(np.arcsin(X2 / np.sqrt(1 - X3**2)))
        # WCC: WHY IS PLUNGE ALWAYS NEGATIVE???????
        plunge = np.degrees(np.arcsin(-X3))
        if Y3 == 0:
            rotation = 0
        else:
            rotation = np.degrees(np.arcsin(Y3 / np.sqrt(1 - X3**2)))
        if debug:
            print(azimuth, plunge, rotation)
        return azimuth, plunge, rotation

    def __to_eigen(self, debug=False):
        """Return eigenvector matrix corresponding to ellipsoid

        Internal because x, y and z are in ConfidenceEllipsoid order

        WCC: ARE YOU SURE?  I THOUGHT THEY WERE y, x, z ORDER?"""
        eigvals = (self.semi_major_axis_length**2,
                   self.semi_minor_axis_length**2,
                   self.semi_intermediate_axis_length**2)
        # https://math.stackexchange.com/questions/1403126/what-is-the-general-equation-equation-for-rotated-ellipsoid
        RZ_azi = Ellipsoid.__ROT_RH_azi(np.radians(self.major_axis_azimuth))
        RY_plunge = Ellipsoid.__ROT_RH_plunge(np.radians(
            self.major_axis_plunge))
        RX_rot = Ellipsoid.__ROT_RH_rot(np.radians(self.major_axis_rotation))

        # eigvecs = np.matmul(RZ_azi , RY_plunge , RX_rot)
        eigvecs = RZ_azi * RY_plunge * RX_rot

        # r = self.__rotmat()
        # eigvecs  = r.as_dcm()
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
        """
        Right handed rotation matrix for "azimuth" in RADIANS
        """
        c_azi, s_azi = np.cos(azi), np.sin(azi)
        r = R.from_dcm([[c_azi, -s_azi, 0],
                        [s_azi, c_azi, 0],
                        [0, 0, 1]])
        RX = r.as_dcm()
        return RX
        # return R.from_euler('z', azi).as_dcm()  # WCC: returns the same result?

    @staticmethod
    def __ROT_RH_plunge(plunge):
        """
        Right handed rotation matrix for "plunge" in RADIANS
        """
        c_plunge, s_plunge = np.cos(plunge), np.sin(plunge)
        r = R.from_dcm([[c_plunge, 0, s_plunge],
                       [0, 1, 0],
                       [-s_plunge, 0, c_plunge]])
        RY = r.as_dcm()
        return RY
        # return R.from_euler('x', plunge).as_dcm()  # WCC: returns the same result?

    @staticmethod
    def __ROT_RH_rot(rot):
        """
        Right handed rotation matrix for "rotation" in RADIANS
        """
        c_rot, s_rot = np.cos(rot), np.sin(rot)
        r = R.from_dcm([[1, 0, 0],
                       [0, c_rot, -s_rot],
                       [0, s_rot, c_rot]])
        RZ = r.as_dcm()
        return RZ
        # return R.from_euler('y', rot).as_dcm()  # WCC: returns the same result?

    def to_covariance(self, debug=False):
        """
        Return covariance matrix corresponding to ellipsoid

        Uses eigenvals * cov = eigenvecs * cov
        """
        eigvals, eigvecs = self.__to_eigen()
        cov = eigvecs * np.diag(eigvals) * np.linalg.inv(eigvecs)
        return cov

    def to_XYEllipse(self, debug=False):
        """
        Return XY-ellipse corresponding to Ellipsoid

        Should probably make a generic to_Ellipse(), allowing one
        to extract the Ellipse viewed from any angle. to_XYEllipse()
        would call to_Ellipse() from the appropriate view angle
        """
        cov = self.to_covariance()
        # print(cov)
        errors = np.sqrt(np.diag(cov))
        cross_covs = cov[0, 1], cov[0, 2], cov[1, 2]
        cov_xy = [[errors[0]**2, cross_covs[0]],
                  [cross_covs[0], errors[1]**2]]
        # cov_xy = cov[0:1,0:1]  # WCC: MORE COMPACT, SAME ANSWER?

        evals, evecs = np.linalg.eig(cov_xy)
        sort_indices = np.argsort(evals)[::-1]
        a, b = np.sqrt(evals[sort_indices[0]]), np.sqrt(evals[sort_indices[1]])
        x_v1, y_v1 = evecs[:, 0]
        if y_v1 == 0.:
            theta = 90.
        else:
            theta = (np.degrees(np.arctan((x_v1) / (y_v1))) + 180) % 180
        return a, b, theta

    # def to_XYEllipse(self, debug=False):
    #    """Return XY-plane Ellipse corresponding to Ellipsoid
    #
    #    Should probably make a generic to_Ellipse, allowing one
    #    to extract the Ellipse viewed from any angle, then to_XYEllipse
    #    would call this code from the appropriate view angle
    #    """
    #    print('to_XYEllipse is not yet written!')

    def to_uncerts(self, debug=False):
        """
        Return errors and covariances corresponding to ellipsoid
        
        :returns errors, cross_covs:
        :rtype errors: 3-tuple of xerr, yerr, zerr errors
        :rtype cross_covs: 3-tuple of c_xy, c_xz, c_yz
        """
        cov = self.to_covariance()
        errors = np.sqrt(np.diag(cov))
        cross_covs = cov[0, 1], cov[0, 2], cov[1, 2]
        return errors, cross_covs

    def check_equi_covariance(self, debug=False):
        """
        check equaivalnce between from/to covariance
        """
        cov = self.to_covariance()
        ell = self.from_covariance(cov)
        # print(ell)
        # assert np.all(ell() == self.()), 'not equal'
        return ell

    def check_equi_uncerts(self, debug=False):
        """
        check equivalence between from/to uncertainties
        """
        errors, cross_covs = self.to_uncerts()
        el = self.from_uncerts(errors, cross_covs)
        return el

    def plot(self, title=None, debug=False):
        """
        Plots ellipsoid viewed from -z, corresponding to view from above

        https://stackoverflow.com/questions/7819498/plotting-ellipsoid-
              with-matplotlib
        """
        # Make set of spherical angles to draw our ellipsoid
        n_points = 100
        theta = np.linspace(0, 2 * np.pi, n_points)
        phi = np.linspace(0, np.pi, n_points)

        # Get the xyz points for plotting
        # Cartesian coordinates that correspond to the spherical angles:
        X = self.semi_minor_axis_length * np.outer(np.cos(theta), np.sin(phi))
        Y = self.semi_major_axis_length * np.outer(np.sin(theta), np.sin(phi))
        Z = self.semi_intermediate_axis_length *\
            np.outer(np.ones(np.size(theta)), np.cos(phi))

        old_shape = X.shape
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        r = self.__rotmat()
        # Rotate ellipsoid
        XYZ_rot = r.apply(np.array([X, Y, Z]).T)
        X_rot, Y_rot, Z_rot = (XYZ_rot[:, 0].reshape(old_shape),
                               XYZ_rot[:, 1].reshape(old_shape),
                               XYZ_rot[:, 2].reshape(old_shape))
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X_rot, Y_rot, Z_rot, alpha=0.3, color='r')
        ax.view_init(elev=-140., azim=-55.)
        #  x,y of ellipse is not same as x,y of covariance matrix.
        # Swapped because x and y correspond to N and E respectively in
        # covariance matrix.
        ax.set_xlabel('y(E)')
        ax.set_ylabel('x(N)')
        ax.set_zlabel('z(Depth)')
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