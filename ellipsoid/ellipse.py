#! /bin/env python3
import numpy as np
import math as m
import matplotlib.pyplot as plt
import copy

eps=np.finfo(float).eps

class ellipse:
    def __init__(self,a,b,theta=0,center=(0,0)):
        """Defines an ellipse
        
        The ellipse is assumed to be centered at zero with its semi-major axis aligned
        along the x-axis unless orientation and/or center are set otherwise
        
        Inputs:
            a=length of semi-major axis
            b=length of semi-minor axis
            theta=orientation (radians CCW from 0=semi-major along x-axis)
            center=x,y coordinates of ellipse center (0,0)
        """
        if a < b:
            print('Your semi-major is smaller than your semi-minor!  Switching...')
            self.a=b
            self.b=a
        else:
            self.a=a
            self.b=b
        self.theta=theta
        self.x=center[0]
        self.y=center[1]
        
    @classmethod
    def from_uncerts(cls, x_err, y_err, c_xy, center=(0,0),debug=False):
        """Set error ellipse using common epicenter uncertainties
        
        Call as e=ellipse.from_uncerts(x_err,y_err,c_xy,center)
        
        Inputs:
            x_err: x error (m)
            y_err: y error (m)
            c_xy:  x-y covariance (m^2)
            center:  center position (x,y)
            
        From http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        and https://blogs.sas.com/content/iml/2014/07/23/prediction-ellipses-from-covariance.html
        """
        cov=[[x_err**2, c_xy],[c_xy, y_err**2]]
        evals, evecs = np.linalg.eig(cov)
        if debug:
            print(evecs)
            print(evals)
        # Sort eigenvalues in decreasing order and select the semi-major and semi-minor axis lengths
        sort_indices = np.argsort(evals)[::-1]
        a,b=evals[sort_indices[0]] , evals[sort_indices[1]]
        # Calculate angle of semi-major axis
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        theta = np.arctan((y_v1)/(x_v1))  
        return cls(a,b,theta,center)

    @classmethod
    def from_uncerts_baz(cls, x_err, y_err, c_xy, dist, baz,viewpoint=(0,0),debug=False):
        """Set error ellipse using common epicenter uncertainties, back-azimuth
        
        Call as e=ellipse.from_uncerts_baz(x_err,y_err,c_xy,dist,baz[,viewpoint])
        
        Inputs:
            x_err: x error (m)
            y_err: y error (m)
            c_xy:  x-y covariance (m^2)
            dist:  distance of center from observer
            baz:   back-azimuth from observer (degrees)  
            viewpoint: observer's position [(0,0)]                     
        """
        x = viewpoint[0] + dist*np.sin(np.radians(baz))
        y = viewpoint[1] + dist*np.cos(np.radians(baz))
        return cls.from_uncerts(x_err,y_err,c_xy,(x,y))
        
    def __repr__(self):
        """String describing the ellipse        
        """
        s=f'<a={self.a:8.3g}, b={self.b:8.3g}'
        if self.theta != 0:
            s += f', theta={self.theta:8.3f}'
        if self.x != 0 or self.y != 0:
            s += f', center=({self.x:8.3g},{self.y:8.3g})'
        s+='>'
        return s
        
    @staticmethod
    def __ROT_CCW(theta):
        """counter-clockwise rotation matrix for theta in RADIANS"""
        c,s=np.cos(theta),np.sin(theta)
        return np.array(((c,-s),(s,c))) 

    @staticmethod
    def __ROT_CW(theta):
        """clockwise rotation matrix for theta in RADIANS"""
        c,s=np.cos(theta),np.sin(theta)
        return np.array(((c,s),(-s,c))) 

    def is_inside(self,pt,debug=False):
        """ Is the given point inside the ellipse?
        
        pt = (x,y) coordinates of the point    
        """
        pt1=self.__relative_viewpoint(pt)
        x1,y1=pt1
        value=((x1**2)/(self.a**2)) + ((y1**2)/(self.b**2))
        if debug:
            print(pt1)
            print(value)
        if value <= 1:
            return True
        return False

    def is_on(self,pt):
        """ Is the given point on the ellipse?
        
        pt = (x,y) coordinates of the point    
        """
        pt1=self.__relative_viewpoint(pt)
        x1,y1=pt1
        value=((x1**2)/(self.a**2)) + ((y1**2)/(self.b**2))
        if abs(value - 1) < 2*eps:
            return True
        return False
    
    def __relative_viewpoint(self,pt):
        """ Coordinates of the viewpoint relative to a 'centered' ellipse
    
        A centered ellipse has its center at 0,0 and its semi-major axis
        along the x-axis
    
        Inputs:
            x,y:   coordinates of the viewpoint
    
        Outputs:
            x1,y1: new coordinates of the viewpoint
        """
        # Translate
        pt1=(pt[0]-self.x,pt[1]-self.y)
        
        # Rotate clockwise
        R_rot=ellipse.__ROT_CW(self.theta)
        #=np.cos(-self.theta),np.sin(-self.theta)
        #R_rot=np.array(((c,-s),(s,c))) 
        rotated = np.dot(R_rot,pt1)
        return rotated

    def __absolute_viewpoint(self,pt):
        """ Coordinates of a point after 'uncentering' ellipse
    
        Assume that the ellipse was "centered" for calculations, now
        put the point back in its true position
    
        Inputs:
            x,y:   coordinates of the viewpoint
    
        Outputs:
            x1,y1: new coordinates of the viewpoint
        """
        # Unrotate 
        R_rot=ellipse.__ROT_CCW(self.theta)
        #c,s=np.cos(self.theta),np.sin(self.theta)
        #R_rot=np.array(((c,-s),(s,c))) 
        unrot = np.dot(R_rot,pt)

        # Untranslate
        pt1=(unrot[0]+self.x, unrot[1]+self.y)
        
        return pt1

    def __get_tangents(self,pt=(0,0)):
        """ Return tangent intersections for a point and the ellipse 
        
        
        Equation is from http://www.nabla.hr/Z_MemoHU-029.htm
        P = (-a**2 * m/c, b**2 / c)        
        """
        if self.is_inside(pt):
            print('No tangents, point is inside ellipse')
            return [],[]
        elif self.is_on(pt):
            print('No tangents, point is on ellipse')
            return [],[]
            
        # for calculations, assume ellipse is centered at zero and pointing S-N
        (x1,y1)=self.__relative_viewpoint(pt)        
        ms=np.roots([self.a**2 - x1**2, 2*y1*x1, self.b**2 - y1**2])
        cs= -x1*ms + y1        
        # Determine the tangent intersect with ellipse
        T0=(-self.a**2 *ms[0]/cs[0], self.b**2 /cs[0])
        T1=(-self.a**2 *ms[1]/cs[1], self.b**2 /cs[1])
        
        #Rotate back to true coords
        T0=self.__absolute_viewpoint(T0)
        T1=self.__absolute_viewpoint(T1)
        return T0,T1

    def subtended_angle(self,pt=(0,0),debug=True):
        """ Find the angle subtended by an ellipse when viewed from x,y 
    
        pt=        (x,y) coordinates of the viewpoint
    
        Equations are from http://www.nabla.hr/IA-EllipseAndLine2.htm
        For a "centered" ellipse
            y=mx+c
            a^2*m^2 + b^2 = c^2
            where   x,y are the viewpoint coordinates,
                    a,b are the semi-* axis
                    m and c are unknown
            => (a^2 - x^2)*m^2 + 2*y*x*m + (b^2 - y^2) = 0  [Solve for m]
            and then c  = y - mx
        """
        # If point is on or inside the ellipse, no need to calculate tangents
        if self.is_inside(pt):
            return 2*np.pi
        elif self.is_on(pt):
            return np.pi
        # Move point to origin
        temp=copy.copy(self)
        temp.x-=pt[0]
        temp.y-=pt[1]
        T0,T1=temp.__get_tangents((0,0))
        cosang=np.dot(T0,T1)
        sinang=np.linalg.norm(np.cross(T0,T1))
        return np.arctan2(sinang,cosang)     

    def plot(self):
        """ Plot the ellipse
        """
        t=np.linspace(0,2*np.pi, 100)
        Ell=np.array([self.a*np.cos(t), self.b*np.sin(t)])
        R_rot=ellipse.__ROT_CCW(self.theta)
        #c,s=np.cos(self.theta),np.sin(self.theta)
        #R_rot=np.array(((c,-s),(s,c))) 
        Ell_rot=np.zeros((2,Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
            
        plt.plot(self.x+Ell_rot[0,:], self.y+Ell_rot[1,:],'b')
        plt.axis('equal')
        return plt.gca()
        

    def plot_tangents(self,pt=(0,0)):
        """ Plot tangents to an ellipse when viewed from x,y 
    
        pt=        (x,y) coordinates of the viewpoint
        
        """
        ax=self.plot()
        ax.plot(pt[0],pt[1],'k+')
        T0,T1=self.__get_tangents(pt)
        
        if T0:
            ax.plot([pt[0],T0[0]],[pt[1],T0[1]],'g-')
            ax.plot([pt[0],T1[0]],[pt[1],T1[1]],'g-')
        
        plt.show()
        return