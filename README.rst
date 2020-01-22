===================
Ellipsoid
===================

Class to create confidence ellipses and ellipsoids

***Stupid thing:** some Ellipse creators use (x, y) whereas Ellipsoid uses
(N, E, Z)  

This is because Ellipse was created/coded with x=W and y=N, whereas Ellipsoid
is based on equations in which x=N and y=W (viewed from beneath to allow right-
handedness!).  Ellipse is a natural fit to things coming from SEISAN (x, then y),
whereas Ellipsoid fits with codes in which latitude comes before longitude.
obsPy does not seem to have a consistent policy

- Ellipse.from_covariance(cov, center:
        :param cov: covariance matrix [[c_xx, c_xy], [c_xy, c_yy]]
        :param center: center position (x,y) (m)
- Ellipsoid.from_covariance(cov, center)
        :param cov: covariance matrix [[c_nn, c_ne, c_nz],
                                       [c_en, c_ee, c_ez],
                                       [c_zn, c_ze, c_zz]]
        :param center: center position (N,E,Z) (m)

- Ellipse.from_uncertainties(x_err, y_err, c_xy, center=(0, 0)):
        :param errors: (x, y) errors (m)
        :param c_xy:  x-y cross-covariance (m^2)
        :param center: center position (x,y)
- Ellipsoid.from_uncertainties(errors, cross_covs, center)
        :param errors: (N, E, Z) errors (m)
        :param cross_covs: (c_ne, c_nz, c_ez) cross covariances (m^2)
        :param center: (N, E, Z) center of ellipse]

Already, should change names to .from_covariance and .from_uncertainties

Should the standara creators be 
Ellipsoid(lengths, angles, center)
Ellipse(lengths, azimuth, center)
Instead of
Ellipsoid(semi_major, semi_minor, semi_intermediate, azimuth, plunge, rotation, center)
Ellipse(semi_major, semi_minor, azimuth, center)

where "lengths" and "angles" are tuples?