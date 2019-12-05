#! /bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math as m
import matplotlib.pyplot as plt

eps=np.finfo(float).eps

class ellipsoid:
    def __init__(self, semi_major_axis_length, semi_minor_axis_length,
                 semi_intermediate_axis_length, major_axis_plunge=0,
                 major_axis_azimuth=0, major_axis_rotation=0,
                 center=(0, 0, 0)):
        """Defines an ellipsoid using QuakeML ellipsoid parameters
                
        :param semi_major_axis_length: Largest uncertainty, corresponding to the
            semi-major axis of the confidence ellipsoid. Unit: m
        :param semi_minor_axis_length: Smallest uncertainty, corresponding to the
            semi-minor axis of the confidence ellipsoid. Unit: m
        :param semi_intermediate_axis_length: Uncertainty in direction orthogonal
            to major and minor axes of the confidence ellipsoid. Unit: m
        :param major_axis_plunge: Plunge angle of major axis of confidence
            ellipsoid. Corresponds to Tait-Bryan angle φ. Unit: deg
        :param major_axis_azimuth: Azimuth angle of major axis of confidence
            ellipsoid. Corresponds to Tait-Bryan angle ψ. Unit: deg
        :param major_axis_rotation: This angle describes a rotation about the
            confidence ellipsoid’s major axis which is required to define the
            direction of the ellipsoid’s minor axis. Corresponds to Tait-Bryan
            angle θ.
            
        "center" is only used for plotting, so I don't convert it to
        CenterEllipsoid coordinates
            
        This corresponds to the z-x'-y'' intrinsic rotation scheme (not to be
        confused with z-y'-x'' which is used in airplanes (heading-pitch-roll)
                
        """
        self.semi_major_axis_length = semi_major_axis_length
        self.semi_minor_axis_length = semi_minor_axis_length
        self.semi_intermediate_axis_length = semi_intermediate_axis_length
        self.major_axis_plunge = major_axis_plunge
        self.major_axis_azimuth = major_axis_azimuth
        self.major_axis_rotation = major_axis_rotation
        self.center = center
        self.__error_test()

    def __error_test(self):
        """Test for invalid parameters"""
        # Are axis lengths in the right order (major>intermediate>minor?)
        lengths = [self.semi_minor_axis_length,
                   self.semi_intermediate_axis_length,
                   self.semi_major_axis_length]
        # print(lengths)
        sorted_lengths = np.sort(lengths)
        assert np.all(lengths == sorted_lengths), 'not major > intermed > minor'

    @classmethod
    def from_covariance(cls, cov, center=(0, 0, 0), debug=False):
        """Set error ellipsoid using covariance matrix
        
        Call as e=ellipsoid.from_covariance(cov)
        
        Inputs:
            cov: 3x3 covariance matrix (indices 0,1,2 correspond to x,y,z)
            center: center of the ellipse (0,0,0)
       
        The covariance matric must be symmetric and positive definite
        
        
        From http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        and https://blogs.sas.com/content/iml/2014/07/23/prediction-ellipses-from-covariance.html
        """
        # Check if 3x3 and symmetric
        cov=np.array(cov)
        assert cov.shape == (3,3), 'Covariance matrix is not 3x3'
        assert np.allclose(cov,cov.T,eps), f'Covariance matrix is not symmetri {cov}'
        
        # Transform covariance matrix to CovarianceEllipse coordinates (x=N,y=E)
        temp=cov(0,1)   # old c_xz, becomes c_yz
        cov[0,2],cov[2,0]=cov[1,2],cov[1,2]
        cov[1,2],cov[2,1]=temp,temp
        temp=cov[0,0] # old c_xx, becomes c_yy
        cov[0,0]=cov[1,1]
        cov[1,1]=temp
        
        # EIGH() returns eig fast and sorted if input matrix symmetric
        evals, evecs = np.linalg.eigh(cov) 
        
        assert np.all(evals>0), 'Covariance matrix is not positive definite'
        assert np.allclose(np.linalg.norm(evecs),[1.,1.,1.]), 'Eigenvectors are not unit length'
        
        if debug:
            print(evecs)
            print(evals)
        
        # Semi-major axis lengths
        s_min,s_inter,s_maj=np.sqrt(evals)
        
        # Calculate angles of semi-major axis
        # From wikipedia (z-x'-y'' convention, left-hand rule, x=, x=E, z=UP)
        x_min, y_min, z_min = evecs[:, 0]  # Unit semi-minor axis ("Y")
        x_maj, y_maj, z_maj = evecs[:, 2]  # Unit semi-major axis ("X")
        azimuth=  np.degrees(np.arcsin(y_maj/np.sqrt(1-z_maj**2)))
        plunge=  np.degrees(np.arcsin(-z_maj))
        rotation=np.degrees((z_min/np.sqrt(1-z_maj**2)))
        return cls(s_maj,s_min,s_inter,plunge,azimuth,rotation,center)
        
    @classmethod
    def from_uncerts(cls, errors, cross_covs=(0,0,0), center=(0,0,0),debug=False):
        """Set error ellipse using common epicenter uncertainties
        
        Call as e=ellipsoid.from_uncerts(errors, cross_covs, center)
        
        x is assumed to be Latitudes, y Longitudes
        
        Inputs:
            errors:      (x, y, z) errors (m)
            cross_covs:  (c_xy, c_xz, c_yz) covariances (m^2) [(0,0,0)]
            center:      (x, y, z) center of ellipse [(0,0,0)]
        """
        cov=[[errors[0]**2,  cross_covs[0], cross_covs[1]],
             [cross_covs[0], errors[1]**2,  cross_covs[2]],
             [cross_covs[1], cross_covs[2], errors[2]**2]]
        return cls.from_uncerts(cov,center)
        
    def __to_eigen(self,debug=False):
        """Return eigenvector matrix corresponding to ellipsoid 
        
        Internal because x,y and z are in ConfidenceEllipsoid order"""

        eigvals=(self.semi_major_axis_length,self.semi_minor_axis_length,self.semi_intermediate_axis_length)
        # Use notation and formulats from wikipedia       
        azi = np.radians(self.major_axis_azimuthh)
        plunge =   np.radians(self.major_axis_plunge)
        rot =   np.radians(self.major_axis_rotation)
        # Use wikipedia notation
        c_azi,s_azi =        np.cos(azi),   np.sin(azi)
        c_plunge,s_plunge =  np.cos(plunge),np.sin(plunge)
        c_rot,s_rot =        np.cos(rot),   np.sin(rot)
        # Currently right-handed
        RZ=np.array([[   1,       0,        0   ],
                     [   0,     c_azi,   -s_azi],
                     [   0,     s_azi,    c_azi]])
        RY=np.array([[c_plunge,    0,     s_plunge],
                     [0,           1,        0 ],
                     [-s_plunge,   0,     c_plunge]])
        RX=np.array([[c_rot,     -s_rot,     0 ],
                     [s_rot,      c_rot,     0 ],
                     [  0  ,       0,      c_rot]])
        eigvecs=RZ*RY*RX
        
        return eigvals,eigvecs
        
    def to_covariance(self,debug=False):
        """Return covariance matrix corresponding to ellipsoid 
        
        Uses eigenvals*cov=eigenvecs*cov
        """
        
        eigvals,eigvecs=self.__to_eigen()
        cov=eigenvecs*np.diag(eigvals)*np.linalg.inv(eigvecs)
        
        # Convert to standard coordinates (x=E, y=N)
        # Transform covariance matrix to CovarianceEllipse coordinates (x=N,y=E)
        temp=cov(0,1)   # old c_xz, becomes c_yz
        cov[0,2],cov[2,0]=cov[1,2],cov[1,2]
        cov[1,2],cov[2,1]=temp,temp
        temp=cov[0,0] # old c_xx, becomes c_yy
        cov[0,0]=cov[1,1]
        cov[1,1]=temp
        
        return cov
        
    def to_xyz(self,debug=False):
        """Return xyz and covariances corresponding to ellipsoid """
        cov=self.to_covariance()
        errors=np.sqrt(np.diag(cov))
        cross_covs=cov[0,1],cov[0,2],cov[1,2]
        return errors,cross_covs
        

    def plot(self,debug=False):
        """Plots ellipsoid """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Make set of spherical angles to draw our ellipsoid
        n_points = 100
        theta = np.linspace(0, 2*np.pi, n_points)
        phi = np.linspace(0, np.pi, n_points)

        # Get ellipsoid parameters
        eigvals, eigvecs=self.__to_eigen()
        
        # Width, height and depth of ellipsoid
        rx, ry, rz = np.sqrt(eigvals)

        # Get the xyz points for plotting
        # Cartesian coordinates that correspond to the spherical angles:
        X = rx * np.outer(np.cos(theta), np.sin(phi))
        Y = ry * np.outer(np.sin(theta), np.sin(phi))
        Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

        # Rotate ellipsoid
        old_shape = X.shape
        X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
        X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
        X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
        # Add in offsets, flipping X and Y to correspond to "external" convention
        X = Y + self.center[0]
        Y = X + self.center[1]
        Z = Z + self.center[2]
        
        # Plot
        ax.plot_wireframe(X,Y,Z, color='r', alpha=0.1)
        # plt.set_aspect(1.0)
        plt.xlabel('x(E)')
        plt.ylabel('y(N)')
        

    def __repr__(self):
        """String describing the ellipsoid        
        """
        s=f'<semi_major={self.semi_major_axis_length:8.3g}, semi_minor={self.semi_minor_axis_length:8.3g}, semi_intermediate_={self.semi_intermediate_axis_length:8.3g}'
        if not np.all([self.major_axis_plunge,self.major_axis_azimuth,self.major_axis_rotation] == 0):
            s += f', major axis plunge,azimuth,rotation=({self.major_axis_plunge:8.3f},{self.major_axis_azimuth:8.3f},{self.major_axis_rotation:8.3f})'
        if not np.all(self.center==0):
            s += f', center=({self.center[0]:8.3g},{self.center[1]:8.3g},{self.center[2]:8.3g})'
        s+='>'
        return s
        


# THIS IS FROM GITHUB circusmonkey/covariance-ellipsoid       
# def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
#     """
#     Return the 3d points representing the covariance matrix
#     cov centred at mu and scaled by the factor nstd.
#     Plot on your favourite 3d axis. 
#     Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
#     Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
#     """
#     assert cov.shape==(3,3)
# 
#     Find and sort eigenvalues to correspond to the covariance matrix
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     idx = np.sum(cov,axis=0).argsort()
#     eigvals_temp = eigvals[idx]
#     idx = eigvals_temp.argsort()
#     eigvals = eigvals[idx]
#     eigvecs = eigvecs[:,idx]
# 
#     Set of all spherical angles to draw our ellipsoid
#     n_points = 100
#     theta = np.linspace(0, 2*np.pi, n_points)
#     phi = np.linspace(0, np.pi, n_points)
# 
#     Width, height and depth of ellipsoid
#     rx, ry, rz = nstd * np.sqrt(eigvals)
# 
#     Get the xyz points for plotting
#     Cartesian coordinates that correspond to the spherical angles:
#     X = rx * np.outer(np.cos(theta), np.sin(phi))
#     Y = ry * np.outer(np.sin(theta), np.sin(phi))
#     Z = rz * np.outer(np.ones_like(theta), np.cos(phi))
# 
#     Rotate ellipsoid for off axis alignment
#     old_shape = X.shape
#     Flatten to vectorise rotation
#     X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
#     X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
#     X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
#    
#     Add in offsets for the mean
#     X = X + mu[0]
#     Y = Y + mu[1]
#     Z = Z + mu[2]
#     
#     return X,Y,Z



#########################################################
## Testing some data
#########################################################
if __name__=='__main__':

    #########################################################
    ## Generate datasets s1 and s2
    #########################################################
    # s1
    mu1 = np.random.random((3)) * 5
    cov1 = np.array([            # Using an off-axis alignment for s1
        [2.5,   0.75,  0.175],
        [0.75,  0.70,  0.135],
        [0.175, 0.135, 0.43]
    ])
    #s1 = np.random.multivariate_normal(mu1, cov1, (200))
    #ax.scatter(s1[:,0],s1[:,1],s1[:,2], c='r')

    # s2
    mu2 = np.random.random((3)) * 5 + 4
    cov2 = np.diag((1,3,5))
    #s2 = np.random.multivariate_normal(mu2, cov2, (200))
    #ax.scatter(s2[:,0],s2[:,1],s2[:,2], c='b')

    #########################################################
    ## Process data and plot ellipsoid
    #########################################################
    #nstd = 2    # 95% confidence interval
    # s1
    #mu1_ = np.mean(s1, axis=0)
    #cov1_ = np.cov(s1.T)
    ell1 = ellipsoid.from_covariance(cov1, mu1)
    print(ell1)
    ell1.plot()

    # s2
    #mu2_ = np.mean(s2, axis=0)
    #cov2_ = np.cov(s2.T)
    ell2 = ellipsoid.from_covariance(cov2, mu2)
    print(ell2)
    ell2.plot()

    plt.show()