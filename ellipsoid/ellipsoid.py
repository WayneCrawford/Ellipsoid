#! /bin/env python3
# -*- coding: utf-8 -*-
"""
QuakeML ConfidenceEllipsoid 

Angles are *intrinsic* (about the axes of the rotating coord syst)

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
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.mplot3d import Axes3D
from .ellipse import Ellipse

eps = np.finfo(float).eps
a_max, p_max, r_max = 180., 180., 180.


class Ellipsoid:
    def __init__(self, semi_major_axis_length, semi_minor_axis_length,
                 semi_intermediate_axis_length, azimuth=0,
                 plunge=0, rotation=0,
                 center=(0, 0, 0)):
        """
        Create an Ellipsoid instance
        
        Angles are *intrinsic* (about the axes of the Ellipsoid's coord syst)
        and are applied in the order azimuth, plunge, rotation (Z-Y-X)

        :param semi_major_axis_length: length of the semi-major axis (m)
        :type semi_major_axis_length: float
        :param semi_minor_axis_length: length of the semi-minor axis (m)
        :type semi_minor_axis_length: float
        :param semi_intermediate_axis_length: length of the semi-intermediate axis (m)
        :type semi_intermediate_axis_length: float
        :param azimuth: angle to rotate around Z-axis
        :type azimuth: float
        :param plunge: angle to rotate around Y-axis
        :type plunge: float
        :param rotation: angle to rotate around X-axis
        :type rotation: float
        :param center: center of the Ellipse (N, E, Z)
        :type center: tuple, optional
        :return: Ellipsoid
        :rtype: :class: `~ellipsoid.Ellipsoid`
        """
        self.semi_major = semi_major_axis_length
        self.semi_minor = semi_minor_axis_length
        self.semi_intermediate = semi_intermediate_axis_length
        self.azimuth = azimuth
        self.plunge = plunge
        self.rotation = rotation
        self.center = center
        self._error_test()

    def __repr__(self, as_ints=False):
        """
        String describing the ellipsoid

        :param as_ints: print parameters as integers
        :type as_ints: Boolean
        """
        fmt_code = '{:.3g}'
        if as_ints:
            fmt_code = '{:.0f}'
        fmt_str = 'Ellipsoid({0}, {0}, {0}, {0}, {0}, {0}'.format(fmt_code)
        s = fmt_str.format(
            self.semi_major, self.semi_minor, self.semi_intermediate,
            self.azimuth, self.plunge, self.rotation)
        if np.any(self.center):
            fmt_str = ', {0}, {0}, {0}'.format(fmt_code)
            s += fmt_str.format(
                self.center[0], self.center[1], self.center[2])
        s += ')'
        return s

    def __str__(self, as_ints=False):
        """
        String describing the Ellipsoid

        :param as_ints: print parameters as integers
        :type as_ints: Boolean
        :return: string
        :rtype: str
        """
        return self.__repr__(as_ints)

    def __eq__(self, other, debug=False):
        """
        Returns true if two Ellipsoids are equal
        
        :param other: second Ellipsoid
        :type other:  :class: `~ellipsoid.Ellipsoid`
        :return: equal
        :rtype: bool
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
        # Compare rotation vectors
        self_rmat=self._rotation().as_matrix().round(5)
        other_rmat=other._rotation().as_matrix().round(5)
        if debug:
            print(self_rmat)
            print(other_rmat)
        if np.equal(self_rmat, other_rmat).all():
            return True
        else:
            for fixed_axis in range(2,-1,-1):  # counts down 2, 1, 0...
                self_testmat = _rotmat_fliptwo(self._rotation(), fixed_axis).as_matrix().round(5)
                if  np.equal(self_testmat, other_rmat).all():
                    return True
        return False

    def _error_test(self):
        """
        Test for invalid parameters

        Are axis lengths in the right order (major > intermediate > minor)?
        
        :return: True if right order, False if wrong
        :rtype: bool
        """
        lengths = [self.semi_minor, self.semi_intermediate, self.semi_major]
        # print(lengths)
        sorted_lengths = np.sort(lengths)
        assert np.all(lengths == sorted_lengths),\
            'not major > intermed > minor'

    @classmethod
    def from_covariance(cls, cov, center=(0, 0, 0), debug=False):
        """Set error ellipsoid using covariance matrix

        :param cov: covariance matrix (0, 1, 2 correspond to N, E, Z)
        :type cov: numpy.array or list of lists
        :param center: center of the ellipse (N,E,Z)
        :type center: tuple, optional
        :return: Ellipsoid
        :rtype: :class: `~ellipsoid.Ellipsoid`

        The covariance matrix must be symmetric and positive definite
        
        Sets azimuth, plunge and rotation within the bounds [0, 180[
        (covers all possible ellipsoids, since ellipsoids are symmetric
        around their principal axes)

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
        s_min, s_inter, s_maj, rotmat = Ellipsoid._eigen_to_rot(evals, evecs)

        azi, plunge, rot = cls._calc_rotation_angles(rotmat)
        obj = cls(s_maj, s_min, s_inter, azi, plunge, rot, center)
        if debug:
            np.set_printoptions(precision=2, suppress=True)
            print('--- from_covariance ---')
            print('cov = {}'.format(
                np.array2string(cov, separator=",", prefix=6*' ')))
            print('rotmat = {}'.format(
                np.array2string(rotmat, separator=",", prefix=9*' ')))
            print('rot.as_matrix = {}'.format(
                np.array2string(obj._rotation().as_matrix(),
                                separator=",", prefix=16*' ')))
            print('rotmat/rot.as_matrix = {}'.format(
                np.array2string(np.divide(rotmat,obj._rotation().as_matrix()),
                                separator=",", prefix=23*' ')))
        return obj

    @staticmethod
    def _eigen_to_rot(evals, evecs, debug=False):
        """
        Create rotation matrix corresponding to eigenvalues/vectors
        
        Doesn't always return the same axis directions as the input covariance,
        but this is handled in _calc_rotation_angles()
    
        :param evals: eigenvalues
        :param evecs: matrix of eigenvectors (columns)
        """
        i_sort = np.argsort(evals)
        evals = evals[i_sort]
        evecs = evecs[:, i_sort]
        imin, iint, imaj = 0, 1, 2
        i_axorder = [imaj, imin, iint]

        # Force right-hand rule
        if np.dot(evecs[:, iint], 
                  np.cross(evecs[:, imaj], evecs[:, imin])) < 0:
            evecs[:, iint] *= -1

        if debug:
            np.set_printoptions(precision=2, suppress=True)
            print('===from_covariance()._eigen_to_rot===')
            print('evecs = {}'.format(
                np.array2string(evecs[:, i_axorder],
                                separator=",", prefix=8*' ')))
            print('evecs[:, iint] . (evecs[:, imaj] x evecs[:, imin]) = {:g}'.\
                format(np.dot(evecs[:, iint],
                              np.cross(evecs[:,imaj], evecs[:,imin]))))

        s_min, s_inter, s_maj = np.sqrt(evals[[imin, iint, imaj]])
        return s_min, s_inter, s_maj, evecs[:, i_axorder]

    @staticmethod
    def _calc_rotation_angles(evecs, debug=False):
        """
        Calculate rotation angles from eigenvectors

        :param evecs: eigenvector matrix, ordered from semi-minor (column 0)
                      to semi-major (column 3)
        """
        rot = R.from_matrix(evecs) 
        azi, plunge, rotation = _get_ZYX_angles(rot)
        if not _are_valid_angles(azi, plunge, rotation):
            if debug:
                print('Handling invalid angles')
            azi, plunge, rotation = _find_valid_angles(azi, plunge, rotation)
            for fixed_axis in range(2,-1,-1):  # counts down 2, 1, 0...
                if azi is not None:
                    break
                azi, plunge, rotation = _get_ZYX_angles(
                    _rotmat_fliptwo(rot, fixed_axis))
                azi, plunge, rotation = _find_valid_angles(azi,
                                                           plunge, rotation)
            assert _are_valid_angles(azi, plunge, rotation)                
        return (azi, plunge, rotation)

    @classmethod
    def from_uncerts(cls, errors, cross_covs=(0, 0, 0), center=(0, 0, 0),
                     debug=False):
        """Set error ellipse using common epicenter uncertainties

        Call as e=ellipsoid.from_uncerts(errors, cross_covs, center)

        N, E, Z = Depth

        Inputs:
        :param errors: (N, E, Z) errors (m)
        :type errors: tuple
        :param cross_covs: (c_NE, c_NZ, c_EZ) covariances (m^2) [(0,0,0)]
        :type cross_covs: tuple, optional
        :param center: (N, E, Z) center of ellipse [(0,0,0)]
        :type center: tuple, optional
        :return: Ellipsoid
        :rtype: :class: `~ellipsoid.Ellipsoid`
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
        eigvals = (self.semi_major**2, self.semi_minor**2,
                   self.semi_intermediate**2)
        # ARE THE EIGENVECTORS REALLY JUST THE ROTATION MATRIX?
        # TRIED self._rotaion().inv(): no dice
        eigvecs = self._rotation().as_matrix()
        cov = np.matmul(np.matmul(eigvecs, np.diag(eigvals)),
                        np.linalg.inv(eigvecs))
        if debug:
            np.set_printoptions(precision=2, suppress=True)
            print('--- to_covariance() ---')
            print("Ellipsoid._rotation().as_matrix() = {}".format(
                 np.array2string(self._rotation().as_matrix(),
                                 separator=",", prefix=36*' ')))
            print('cov = {}'.format(
                np.array2string(cov, separator=",", prefix=6*' ')))

        return cov

    def _rotation(self):
        """
        Return Ellipsoid's rotation
        
        :returns: Ellipsoids rotation object
        :rtype: :class: `~scipy.spatial.transform.Rotation`
        """
        rot = R.from_euler('ZYX',
                           (self.azimuth, self.plunge, self.rotation),
                           degrees=True)
        return rot

    def to_Ellipse(self):
        """
        Return the Ellipse bounding the Ellipsoid, viewed from above

        Should probably also create to_Ellipse_ZN() and to_Ellipse_ZE()
        (for side views)
        :returns: Ellipse
        :rtype: :class: `~ellipsoid.Ellipse`
        """
        cov = self.to_covariance()
        # our Y corresponds to Ellipse's X and vice versa!
        return Ellipse.from_cov([[cov[1, 1], cov[0, 1]],
                                [cov[1, 0], cov[0, 0]]])

    def to_uncerts(self, debug=False):
        """
        Return errors and covariances corresponding to ellipsoid

        :returns errors, cross_covs:
        :rtype errors: 3-tuple of xerr, yerr, zerr errors
        :rtype cross_covs: 3-tuple of c_ne, c_nz, c_xz
        """
        cov = self.to_covariance()
        errors = np.sqrt(np.diag(cov))
        cross_covs = cov[0, 1], cov[0, 2], cov[1, 2]
        return errors, cross_covs

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
            the output format. If no format is found, defaults to png.
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
        r = self._rotation()
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
    """Make all axes equal length in a 3D plot"""
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


def _get_ZYX_angles(rot):
    """ Return the Z-Y-X sequence Tait-Bryan angles of a rotation matrix
    
    Corrects angles very near to (but not at) boundary values
    
    :param rot: rotation matrix
    :type rot: :class: `~scipy.Rotation`
    :return: azimuth, plunge, rotation
    :rtype: tuple
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=UserWarning)
        # warnings.simplefilter("ignore")
        azi, plunge, rotation = rot.as_euler('ZYX', degrees=True)
    azi, plunge, rotation = _correct_angle_numerrs(azi, plunge, rotation)
    return azi, plunge, rotation

    
def _rotmat_fliptwo(rot, fixed_axis=2, debug=False):
    """ Flip two axes in a rotation matrix (retains handedness)
    
    :param rot: 3-dimensional rotation matrix
    :type rot: :class: `~numpy.array()`
    :param fixed_axis: index of the fixed axis (0, 1, or 2)
    :type fixed_axis: int, optional
    """
    rotmat = rot.as_matrix()
    if debug:
        print(f'fliptwo_{fixed_axis}_in: {rotmat}')
    if fixed_axis == 0:
        rotmat[:, 1:] *= -1.
    elif fixed_axis == 1:
        rotmat[:, [0, 2]] *= -1.
    else:
        rotmat[:, :2] *= -1.
    if debug:
        print(f'fliptwo_{fixed_axis}_out: {rotmat}')
    return R.from_matrix(rotmat)


def _are_valid_angles(azi, plunge, rotation):
    """True if angles are in expected range
    
    :param azi: azimuth (0. <= azi < 180.)
    :type azi: float
    :param plunge: plunge (0. <= plunge < 180.)
    :type plunge: float
    :param rotation: rotation (0. <= rotation <= 90.)
    :type rotation: float
    :return: True or False
    :rtype: bool
    """
    if azi is None:
        return False  
    if azi < 0 or azi >= a_max:
        return False
    if plunge < 0 or plunge >= p_max:
        return False
    # if rotation < 0 or rotation > 90:
    if rotation < 0 or rotation >= r_max:
        return False
    return True


def _find_valid_angles(azi, plunge, rotation, debug=False):
    """ Return an equivalent Ellipsoid with valid angles, if it exists
    
    :param azi: azimuth (0. <= azi < 180.)
    :type azi: float
    :param plunge: plunge (0. <= plunge < 180.)
    :type plunge: float
    :param rotation: rotation (0. <= rotation <= 180.)
    :type rotation: float
    :return: azimuth, plunge, rotation if valid ones found
             None, None, None if not
    :rtype: tuple
    """
    azi, plunge, rotation = _correct_angle_numerrs(azi, plunge, rotation)
    if debug:
        print(f'start azi, plunge, rot = {azi}, {plunge}, {rotation}')
    if azi < 0 or azi >= a_max:
        azi = (azi + 180) % 360
        # If flip azimuth, have to invert plunge and rotation
        plunge *= -1
        rotation *= -1

    if plunge < 0 or plunge >= p_max:
        plunge = (plunge + 180) % 360
        # If flip plunge, have to invert rotation
        rotation *= -1.

    if rotation < 0 or rotation >= r_max:
        rotation = (rotation + 180) % 360        
        if rotation < 0 or rotation >= r_max:
            return None, None, None
    if debug:
        print(f'end azi, plunge, rot = {azi}, {plunge}, {rotation}')     
    azi, plunge, rotation = _correct_angle_numerrs(azi, plunge, rotation)
    return azi, plunge, rotation


def _correct_angle_numerrs(azi, plunge, rotation, eps=1e-5):
    """Correct angles outside of allowed range by numerical error only"""
    azi = _correct_close(azi, 0., eps)
    azi = _correct_close(azi, a_max, eps)
    plunge = _correct_close(plunge, 0., eps)
    plunge = _correct_close(plunge, p_max, eps)
    rotation = _correct_close(rotation, 0., eps)
    rotation = _correct_close(rotation, r_max, eps)
    rotation = _correct_close(rotation, -180., eps)
    return azi, plunge, rotation

    
def _correct_close(value, target, eps=1e-5):
    """ Move value to target if it is closer than eps """
    if np.abs(value - target) < eps:
        return target
    return value
    """Correct errors where angles outside of range by numerical error only"""


