===================
Ellipsoid
===================

Class to create confidence ellipses and ellipsoids

Stupid thing: some Ellipse and Ellipse creators are inverted with respect to
each other: Ellipse as x, y, Ellipsoid as N, E.  Reason is that Ellipse was
created/coded with x=W and y=N, but Ellipsoid was based on equations in which
x=N and y=W.  Ellipse is a natural fit to things coming from SEISAN (x, then y).
Ellipsoid fits with codes in which latitude comes before longitude.  obsPy flits
between both

- Ellipse.from_cov(cov, center:
        :param cov: covariance matrix [[c_xx, c_xy], [c_xy, c_yy]]
        :param center: center position (x,y)
- Ellipse.from_uncerts(x_err, y_err, c_xy, center=(0, 0)):
        :param x_err: x error (m)
        :param y_err: y error (m)
        :param c_xy:  x-y cross-covariance (m^2)
        :param center: center position (x,y)


Ellipsoids are created using:

- Ellipsoid.from_covariance(cov, center)
        :param cov: covariance matrix [[c_nn, c_ne, c_nz],
                                       [c_en, c_ee, c_ez],
                                       [c_zn, c_ze, c_zz]]
        :param center: center of the ellipse (N,E,Z)
- Ellipse.from_uncerts(errors, cross_covs, center)
        :param errors: (N, E, Z) errors (m)
        :param cross_covs: (c_ne, c_nz, c_ez) covariances (m^2)
        :param center: (N, E, Z) center of ellipse]

Already, should change names to .from_covariance and .from_uncertainties


Should the creators be 
Ellipsoid (lengths, angles, center)
Ellipse (lengths, azimuth, center)
Instead of
Ellipsoid(semi_major, semi_minor, semi_intermediate, azimuth, plunge, rotation, center)
Ellipse(semi_major, semi_minor, azimuth, center)
?