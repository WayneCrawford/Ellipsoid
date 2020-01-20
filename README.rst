===================
Ellipsoid
===================

Class to create confidence ellipses and ellipsoids

One stupid thing: some Ellipse and Ellipse creators are inverted with respect to each other :
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
        :param cov: covariance matrix (0, 1, 2 correspond to N, E, Z)
        :param center: center of the ellipse (N,E,Z)
- Ellipse.from_uncerts(errors, cross_covs, center)
        :param errors: (N, E, Z) errors (m)
        :param cross_covs: (c_NE, c_NZ, c_EZ) covariances (m^2)
        :param center: (N, E, Z) center of ellipse]

Already, should change names to .from_covariance and .from_uncertainties
