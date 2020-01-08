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
        center = (N, E, Z)
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
                abs(self.semi_intermediate_)) < 1e-2:
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
        lengths = [self.semi_minor, self.semi_intermediate, self.semi_major]
        # print(lengths)
        sorted_lengths = np.sort(lengths)
        assert np.all(lengths == sorted_lengths),\
            'not major > intermed > minor'

    @classmethod
    def from_covariance(cls, cov, center=(0, 0, 0), debug=False):
        """Set error ellipsoid using covariance matrix

        Call as e=ellipsoid.from_covariance(cov)

        Inputs:
            cov: 3x3 covariance matrix (indices 0, 1, 2 correspond to
                 N, E, Z.
            center: center of the ellipse N,E,Z [= (0,0,0)]

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
        # X1,X2,X3 is not always along the semi major axis, it is along the
        # rotated axis. So if there is xy rotation
        # (in that case cov_xy is not zero) but I have a z axis length
        # greater than rotated x, it will swap x and z and
        # the returned X1,X2,X3 will be worng.
        # for example: cov = 2,2,3,3.99,0,0, eigen values are 2.83,0,3. But
        # eigh() returns in the order 0,2.83,3.
        # If I take simple evecs[;,2] in all cases , Here X3 will be along
        # 3, but it should have along 2.83.
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

        The angles are calculated according to Tait Bryan Wikipedia
        page(https://en.wikipedia.org/wiki/Euler_angles) where plunge was
        negative rotation about y axis. So plunge was negative. But I think
        it is according to the left hand rule whereas we have
        calculated the eigvector matrix according to Right hand rule. So I
        changed the signs of azimuth and plunge. I am not sure
        about rotation angle because its range is (0-180)deg. So it cant be
        negative. But in some cases, it returns negative value.
        Range of azimuth and plunge is (-180 to 180)
        """
        r = R.from_rotvec([
            [0, 0, np.arcsin(-X2 / np.sqrt(1 - X3**2))],
            [0, np.arcsin(X3), 0],
            [np.arcsin(Y3 / np.sqrt(1 - X3**2)), 0, 0]])
        array_angles = r.as_euler('ZYX', degrees=True)
        # print(array_angles)
        if X2 == 0:
            azimuth = 0
        else:
            azimuth = array_angles[0, 0]
        if Y3 == 0:
            rotation = 0
        else:
            rotation = array_angles[2, 2]
        plunge = array_angles[1, 1]
        return azimuth, plunge, rotation

    def __to_eigen(self, debug=False):
        """Return eigenvector matrix corresponding to ellipsoid

        Internal because x, y and z are in ConfidenceEllipsoid order

        WCC: ARE YOU SURE?  I THOUGHT THEY WERE y, x, z ORDER?
        I am not sure. But seems like it is working with this order.
        """
        eigvals = (self.semi_major**2,
                   self.semi_minor**2,
                   self.semi_intermediate**2)
        # https://math.stackexchange.com/questions/1403126/what-is-the-general-equation-equation-for-rotated-ellipsoid
        RZ_azi = Ellipsoid.__ROT_RH_azi(np.radians(self.azimuth))
        RY_plunge = Ellipsoid.__ROT_RH_plunge(np.radians(
            self.plunge))
        RX_rot = Ellipsoid.__ROT_RH_rot(np.radians(self.rotation))

        eigvecs1 = np.matmul(RZ_azi, RY_plunge)
        eigvecs = np.matmul(eigvecs1, RX_rot)

        inv_eigvecs = np.linalg.inv(eigvecs)

        return eigvals, eigvecs, inv_eigvecs

    def __rotmat(self):
        """
        Return rotation matrix of ellipsoid
        ORDER SHOULD BE ZYX I think
        """
#         r = R.from_euler('z', self.azimuth, degrees=True) *\
#             R.from_euler('x', self.plunge, degrees=True) *\
#             R.from_euler('y', self.rotation, degrees=True)
        r = R.from_euler('z', self.azimuth, degrees=True) *\
            R.from_euler('y', self.plunge, degrees=True) *\
            R.from_euler('x', self.rotation, degrees=True)
        return r

    @staticmethod
    def __ROT_RH_azi(azi):
        """
        Right handed rotation matrix for "azimuth" in RADIANS
        """
        return R.from_euler('z', azi).as_dcm()

    @staticmethod
    def __ROT_RH_plunge(plunge):
        """
        Right handed rotation matrix for "plunge" in RADIANS
        """
        return R.from_euler('y', plunge).as_dcm()

    @staticmethod
    def __ROT_RH_rot(rot):
        """
        Right handed rotation matrix for "rotation" in RADIANS
        """
        return R.from_euler('x', rot).as_dcm()

    def to_covariance(self, debug=False):
        """
        Return covariance matrix corresponding to ellipsoid

        Uses eigenvals * cov = eigenvecs * cov
        """
        eigvals, eigvecs, inv_eigvecs = self.__to_eigen()
        # cov = eigvecs * np.diag(eigvals) * np.linalg.inv(eigvecs)
        cov1 = np.matmul(eigvecs, np.diag(eigvals))
        cov = np.matmul(cov1, inv_eigvecs)
        return cov

#     def to_XYEllipse(self, debug=False):
#         """
#         Return XY-ellipse corresponding to Ellipsoid
#         
#         OLD, returns bad value for Ellipse class
#         """
#         cov = self.to_covariance()
#         # print(cov)
#         # errors = np.sqrt(np.diag(cov))
#         # cross_covs = cov[0, 1], cov[0, 2], cov[1, 2]
#         cov_xy = [[cov[0, 0], cov[0, 1]],
#                   [cov[1, 0], cov[1, 1]]]
#         # cov_xy = cov[0:1,0:1]  # WCC: MORE COMPACT, SAME ANSWER?
#         # print(cov_xy)
#         evals, evecs = np.linalg.eig(cov_xy)
#         sort_indices = np.argsort(evals)[::-1]
#         a, b = np.sqrt(evals[sort_indices[0]]), np.sqrt(evals[sort_indices[1]])
#         x_v1, y_v1 = evecs[:, 0]
#         # print(x_v1, y_v1)
#         if y_v1 == 0.:
#             theta = 90.
#         else:
#             theta = (np.degrees(np.arctan((x_v1) / (y_v1))) + 180) % 180
#         return a, b, theta
        
    def to_Ellipse(self, debug=False):
        """
        Return NE-ellipse corresponding to Ellipsoid
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
        theta = np.linspace(0, 2 * np.pi, n_points) # X-Y angle
        phi = np.linspace(0, np.pi, n_points)       # angle from X-Y plane

        # Get the xyz points for plotting
        # Cartesian coordinates that correspond to the spherical angles:
        N = self.semi_minor * np.outer(np.cos(theta), np.sin(phi))
        E = self.semi_major * np.outer(np.sin(theta), np.sin(phi))
        # Process as X and Y for the moment
        X = self.semi_major * np.outer(np.cos(theta), np.sin(phi))
        Y = self.semi_minor * np.outer(np.sin(theta), np.sin(phi))
        Z = self.semi_intermediate *\
            np.outer(np.ones(np.size(theta)), np.cos(phi))

        old_shape = N.shape
        # N, E, Z = N.flatten(), E.flatten(), Z.flatten()
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        r = self.__rotmat()
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
