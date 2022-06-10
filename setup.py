import setuptools

with open("thorondor/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thorondor",
    version="0.3.40",
    description="XANES package using jupyter widgets",
    author="David Simonne and Andrea Martini",
    author_email="david.simonne@synchrotron-soleil.fr, andrea.martini@unito.it",
    url="https://pypi.org/project/thorondor/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],

    keywords = "XANES GUI lmfit widgets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
	include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
    "numpy",
    "pandas",
    "matplotlib",
    "ipywidgets",
    "ipython",
    "scipy",
    "lmfit",
    "emcee",
    "corner",
    "xlrd",
    "numdifftools",
    "tables"]
)