from setuptools import setup
import pathlib 
import os 

version = 0.1
INSTALL_REQUIRES = [
    dvc
    pandas
    numpy
    sklearn
    tensorflow
    mlflow 
    matplotlib
    flask 
    keras
    scikit-image
    opencv-python
]

with open (os.path.join(pathlib.Path(__file__).parent, "README.md")) as f:
    long_description = f.readline()


setup(
    name="api",
    version = version,
    author = "Chloe Ji",
    author_email ="ji.jie@edhec.com",
    license= "MIT",
    description ="",
    long_description = long_description,
    install_requires= INSTALL_REQUIRES,
    packages = find_packages(),
    classifiers= [],
    include_package_data=True,
    package_data = {
        "": ["README.md", "requirements.txt"],
    },
    zip_safe = False
)
