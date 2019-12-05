#! env python3
import numpy as np
from ellipse_obspy import ellipse
import matplotlib.pyplot as plt
#from ellipse import ellipse

#Plot ellipses around a point

x_err,y_err,c_xy=0.7,1.0,0.234e0
#x_err,y_err,c_xy=1.0,1.0,0.0e0
dist=4.0
pt=(4,0)
print(f'Point=({pt[0]:g},{pt[1]:g})')
for baz in range(20,360,60):
    e=ellipse.from_uncerts_baz(x_err,y_err,c_xy,dist,baz,pt)

    sub_ang=e.subtended_angle(pt)
    #print(f'Subtended angle={np.degrees(sub_ang):.1f} degrees')
    print('BAZ={:3.0f}: Ellipse={:55s} Subtended angle={:.1f} degrees'.format(\
                baz,str(e)+',',sub_ang))
    #            baz,str(e)+',',np.degrees(sub_ang)))
    plt.ion()
    e.plot_tangents(pt)
plt.ioff()
plt.show()

#Plot points around an ellipse

x_err,y_err,c_xy=0.3,0.7,0.4e0
center=(0.4,0)
pts=((4,-0.5),(0.5,4),(-4,0.5),(-0.5,-4),(0,0))
for pt in pts:
    #print(f'Point=({pt[0]:g},{pt[1]:g})')
    e=ellipse.from_uncerts(x_err,y_err,c_xy,center)

    sub_ang=e.subtended_angle(pt)
    print('BAZ={:3.0f}: Ellipse={:55s} Subtended angle={:.1f} degrees'.format(\
                baz,str(e)+',',sub_ang))
    plt.ion()
    e.plot_tangents(pt)
plt.ioff()
plt.show()