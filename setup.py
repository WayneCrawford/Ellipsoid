import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()
    
version={}
with open("ellipsoid/version.py") as fp:
    exec(fp.read(),version)

setuptools.setup(
    name="ellipsoid",
    version=version['__version__'],
    author="Wayne Crawford",
    author_email="crawford@ipgp.fr",
    description="Confidence Ellipsoids for seismology",
    long_description=long_description,
    long_description_content_type="text/x-rst; charset=UTF-8",
    url="https://github.com/WayneCrawford/ellipsoid",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
          'obspy>=1.1',
          'numpy>=1.17',
          'scipy>=1.4',
          'matplotlib>3.0'
      ],
    python_requires='>=3.6',
    classifiers=(
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics"
    ),
    keywords='seismology'
)