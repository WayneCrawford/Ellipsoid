#! /bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .ellipse import Ellipse

eps = np.finfo(float).eps


class Ellipsoid:
    def __init__(self, semi_major_axis_length, semi_minor_axis_length,
                 semi_intermediate_axis_length, major_axis_azimuth=0,
                 major_axis_plunge=0, major_axis_rotation=0,
                 center=(0, 0, 0)):
        """
        Create an Ellipsoid instance using QuakeML parameterization

        Angles are *intrinsic* (about the axes of the rotating coord syst)
        center = (N, E, Z)

        The QuakeML manual states
        'The three Tait-Bryan rotations are performed as follows:
            (i) a rotation about the Z axis with angle ψ (heading, or azimuth);
            (ii) a rotation about the Y axis with angle φ (elevation, or
                 plunge);
            (iii) a rotation about the X axis with angle θ (bank).
            Note that in the case of Tait-Bryan angles, the rotations are
            performed about the ellipsoid’s axes (X, Y, Z), not about the axes
            of the fixed (x, y, z) Cartesian system.... Note that
            [the x-y geometry]
            can be interpreted as a hypothetical view from the interior of the
            Earth to the inner face of a shell representing Earth’s surface'

        The QuakeML document does not specify the Ellipsoid orientation for
        azimuth, plunge, rotation = 0, 0, 0.  We assume that:
            semi-major is along x (S-N)
            semi-minor is along y (W-E)

        """
        self.semi_major = semi_major_axis_length
        self.semi_minor = semi_minor_axis_length
        self.semi_intermediate = semi_intermediate_axis_length
        self.azimuth = major_axis_azimuth
        self.plunge = major_axis_plunge
        self.rotation = major_axis_rotation
        self.center = center
        self._error_test()

    def __repr__(self, as_ints=False):
        """
        String describing the ellipsoid

        should correspond to command to use to create the ellipsoid
        :parm as_ints: print parameters as integers
        :kind as_ints: Boolean
        """
        fmt_code = '{:.3g}'
        if as_ints:
            fmt_code = '{:.0f}'
        else:
            fmt_code = '{:3g}'
        fmt_str = 'Ellipsoid({0}, {0}, {0}, {0}, {0}, {0}'.format(fmt_code)
        s = fmt_str.format(self.semi_major,
                           self.semi_minor,
                           self.semi_intermediate,
                           self.azimuth,
                           self.plunge,
                           self.rotation)
        if np.any(self.center):
            fmt_str = ', {0}, {0}, {0}'.format(fmt_code)
            s += fmt_str.format(self.center[0],
                                self.center[1],
                                self.center[2])
        s += ')'
        return s

    def __str__(self, as_ints=False):
        """
        String describing the ellipsoid

        should correspond to command to use to create the ellipsoid
        :parm as_ints: print parameters as integers
        :kind as_ints: Boolean
        """
        return self.__repr__(as_ints)

    def __eq__(self, other):
        """
        Returns true if two Ellipsoids are equal
        """
        if not abs((self.semi_major - other.semi_major)
                   / self.semi_major) < 1e-5:
            return False
        if not abs((self.semi_minor - other.semi_minor)
                   / self.semi_minor) < 1e-2:
            return False
        if not abs(
                (self.semi_intermediate - other.semi_intermediate) /
                abs(self.semi_intermediate)) < 1e-2:
            return False
        if not self.center == other.center:
            return False
        if not np.equal(self._rotmat().as_rotvec().round(5),
                        other._rotmat().as_rotvec().round(5)).all():
            return False
        return True

    def _error_test(self):
        """
        Test for invalid parameters

        Are axis lengths in the right order (major > intermediate > minor)?
        """
        lengths = [self.semi_minor, self.semi_intermediate, self.semi_major]
        # print(lengths)
        sorted_lengths = np.sort(lengths)
        assert np.all(lengths == sorted_lengths),\
            'not major > intermed > minor'

    @classmethod
    def from_covariance(cls, cov, center=(0, 0, 0), debug=False):
        """Set error ellipsoid using covariance matrix

        Inputs:
        :param cov: covariance matrix (0, 1, 2 correspond to N, E, Z).
        :param center: center of the ellipse (N,E,Z)

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
        assert np.allclose(np.linalg.norm(evecs, axis=1),[1.,1.,1.]),\
            f'Eigenvectors are not unit length {evecs}'

        s_min, s_inter, s_maj = np.sqrt(evals)  # axis lengths
        azi, plunge, rot = cls._calc_rotation_angles(evecs[:, 2], evecs[:, 0])

        # Semi-major axis lengths
        return cls(s_maj, s_min, s_inter, azi, plunge, rot, center)

    @staticmethod
    def _calc_rotation_angles(X, Y, debug=False):
        """
        Calculate rotation angles from semi-major & -minor eigenvectors

        The angles are calculated using to Tait Bryan Wikipedia
        page(https://en.wikipedia.org/wiki/Euler_angles)

        :param X: semi-major axis eigenvector
        :param Y: semi-minor axis eigenvector
        """
        if X[2] == 1:     # Semi-major axis points down
            plunge = -np.pi/2.
            azimuth = 0
            rotation = np.arcsin(Y[0])
        elif X[2] == -1:   # Semi-major axis points up
            plunge = np.pi/2.
            azimuth = np.arcsin(-Y[0])
            rotation = 0
        else:
            azimuth = np.arcsin(X[1] / np.sqrt(1 - X[2]**2))   # phi equation
            plunge = np.arcsin(-X[2])                          # theta equation
            rotation = np.arcsin(Y[2] / np.sqrt(1 - X[2]**2))  # psi equation
        return np.round(np.degrees((azimuth, plunge, rotation)), 2)

    @classmethod
    def from_uncerts(cls, errors, cross_covs=(0, 0, 0), center=(0, 0, 0),
                     debug=False):
        """Set error ellipse using common epicenter uncertainties

        Call as e=ellipsoid.from_uncerts(errors, cross_covs, center)

        N, E, Z = Depth

        Inputs:
            errors:      (N, E, Z) errors (m)
            cross_covs:  (c_NE, c_NZ, c_EZ) covariances (m^2) [(0,0,0)]
            center:      (N, E, Z?) center of ellipse [(0,0,0)]
        """
        cov = [[errors[0]**2,  cross_covs[0], cross_covs[1]],
               [cross_covs[0], errors[1]**2,  cross_covs[2]],
               [cross_covs[1], cross_covs[2], errors[2]**2]]
        return cls.from_covariance(cov, center)

    def to_covariance(self, debug=False):
        """
        Return covariance matrix corresponding to ellipsoid

        Uses cov = eigvecs * diag(eigvals) * inv(eigvecs) (eqn 15,
        https://www.visiondummy.com/2014/04/
                        geometric-interpretation-covariance-matrix/)
        """
        debug = True
        eigvals, eigvecs = self._to_eigen()
        # cov1 = np.matmul(eigvecs, np.diag(eigvals))
        # cov = np.matmul(cov1, np.linalg.inv(eigvecs))
        cov = np.matmul(np.matmul(eigvecs, np.diag(eigvals)),
                        np.linalg.inv(eigvecs))
        if debug:
            np.set_printoptions(precision=2, suppress=True)
            evals, evecs = np.linalg.eigh(cov)
            evals = evals[[2, 0, 1]]  # Order maj, min, inter
            evecs = evecs[:, [2, 0, 1]]  # Order maj, min, inter
            print(f'eigvals: input={eigvals}, output={evals}')
            e_str = np.array2string(eigvecs, separator=",", prefix=13*' ')
            print(f'inp eigvecs: {e_str}')
            e_str = np.array2string(evecs, separator=",", prefix=13*' ')
            print(f'cov eigvecs: {e_str}')

        return cov

    def _to_eigen(self, debug=False):
        """Return Ellipsoid's eigenvalues and eigenvector matrix

        Internal because x, y and z are in ConfidenceEllipsoid order
        I THINK THERE IS SOMETHING WRONG HERE: THE EIGENVECTORS AREN'T
        JUST THE ROTATION MATRIX?  COULD THEY BE THE INVERSE ROT MATRIX?
        """
        eigvals = (self.semi_major**2,
                   self.semi_minor**2,
                   self.semi_intermediate**2)
        eigvecs = self._rotmat().as_matrix()
        return eigvals, eigvecs

    def _rotmat(self):
        """
        Ellipsoid's rotation matrix
        """
        return R.from_euler(
            'ZYX', (self.azimuth, self.plunge, self.rotation), degrees=True)

    # @staticmethod
    # def _ROT_RH_azi(azi):
    #     """
    #     Right handed rotation matrix for "azimuth" in RADIANS
    #     """
    #     return R.from_euler('z', azi).as_dcm()
    #
    # @staticmethod
    # def _ROT_RH_plunge(plunge):
    #     """
    #     Right handed rotation matrix for "plunge" in RADIANS
    #     """
    #     return R.from_euler('y', plunge).as_dcm()
    #
    # @staticmethod
    # def _ROT_RH_rot(rot):
    #     """
    #     Right handed rotation matrix for "rotation" in RADIANS
    #     """
    #     return R.from_euler('x', rot).as_dcm()

    def to_Ellipse(self, debug=False):
        """
        Return NE-ellipse corresponding to Ellipsoid

        Should probably also create to_Ellipse_ZN() and to_Ellipse_ZE()
        (for side views)
        """
        cov = self.to_covariance()
        # print(cov)
        # errors = np.sqrt(np.diag(cov))
        # cross_covs = cov[0, 1], cov[0, 2], cov[1, 2]
        cov_xy = [[cov[0, 0], cov[0, 1]],
                  [cov[1, 0], cov[1, 1]]]
        # cov_xy = cov[0:1,0:1]  # WCC: MORE COMPACT, SAME ANSWER?
        # print(np.sqrt(cov_xy))
        evals, evecs = np.linalg.eig(cov_xy)
        sort_indices = np.argsort(evals)[::-1]
        a, b = np.sqrt(evals[sort_indices[0]]), np.sqrt(evals[sort_indices[1]])
        x_a, y_a = evecs[:, sort_indices[0]]
        # x_v1, y_v1 = evecs[:, 0]
        # print(x_a, y_a, end = ' ')
        # print(x_v1, y_v1)
        theta = (np.degrees(np.arctan2(x_a, y_a))) % 180
        theta = 90 - theta
        return Ellipse(a, b, theta, self.center([1,0]))

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

    def plot(self, title=None, debug=False, viewpt=(-140, -55),
             outfile=None, format=None, fig=None, show=False):
        """
        Plots ellipsoid viewed from -z, corresponding to view from above

        https://stackoverflow.com/questions/7819498/plotting-ellipsoid-
              with-matplotlib
        :param viewpt: viewpoint (azimuth, elevation)
        :param outfile: Output file string. Also used to automatically
            determine the output format. Supported file formats depend on your
            matplotlib backend. Most backends support png, pdf, ps, eps and
            svg. Defaults to ``None``.
        :param format: Format of the graph picture. If no format is given the
            outfile parameter will be used to try to automatically determine
            the output format. If no format is found it defaults to png output.
            If no outfile is specified but a format is, than a binary
            imagestring will be returned.
            Defaults to ``None``.
        :param fig: Give an existing figure instance to plot into.
            New Figure if set to ``None``.
        :param show: If no outfile/format, sets plt.show()
        """
        # Make set of spherical angles to draw our ellipsoid
        n_points = 50
        theta = np.linspace(0, 2 * np.pi, n_points)  # X-Y angle
        phi = np.linspace(0, np.pi, n_points)        # angle from X-Y plane

        # Get the xyz points for plotting
        # Cartesian coordinates that correspond to the spherical angles:
        # N = self.semi_minor * np.outer(np.cos(theta), np.sin(phi))
        # E = self.semi_major * np.outer(np.sin(theta), np.sin(phi))
        # Process as X and Y for the moment
        X = self.semi_major * np.outer(np.cos(theta), np.sin(phi))
        Y = self.semi_minor * np.outer(np.sin(theta), np.sin(phi))
        Z = self.semi_intermediate *\
            np.outer(np.ones(np.size(theta)), np.cos(phi))

        old_shape = X.shape
        # N, E, Z = N.flatten(), E.flatten(), Z.flatten()
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        r = self._rotmat()
        # Rotate ellipsoid
        # NEZ_rot = r.apply(np.array([N, E, Z]).T)
        XYZ_rot = r.apply(np.array([X, Y, Z]).T)
#         N_rot, E_rot, Z_rot = (NEZ_rot[:, 0].reshape(old_shape),
#                                NEZ_rot[:, 1].reshape(old_shape),
#                                NEZ_rot[:, 2].reshape(old_shape))
        N_rot = -XYZ_rot[:, 0].reshape(old_shape) + self.center[0]
        E_rot = -XYZ_rot[:, 1].reshape(old_shape) + self.center[1]
        Z_rot = -XYZ_rot[:, 2].reshape(old_shape) + self.center[2]
        # Plot
        if not fig:
            fig = plt.figure(figsize=(3, 3), dpi=200)
            fig.add_subplot(111, projection='3d')
        ax = fig.gca()
        # ax.view_init(azim=viewpt[0], elev=viewpt[1])
        # change viewpoint to geographical coordinates (azim=azim-90)
        ax.view_init(azim=90-viewpt[0], elev=viewpt[1])
        ax.plot_wireframe(E_rot, N_rot, Z_rot, alpha=0.3, color='r')
        ax.set_xlabel('E')
        ax.set_ylabel('N')
        ax.set_zlabel('Z(Depth)')
        if title:
            plt.title(title)
        _set_axes_equal(ax)
        if outfile:
            if format:
                fig.savefig(outfile, dpi=100, transparent=True, format=format)
            else:
                fig.savefig(outfile, dpi=100, transparent=True)
        elif format and not outfile:
            imgdata = io.BytesIO()
            fig.savefig(imgdata, format=format, dpi=100, transparent=True)
            imgdata.seek(0)
            return imgdata.read()
        else:
            if show:
                plt.show()
            return fig


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
