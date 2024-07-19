import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyNeVer",
    version="1.0.0",
    author="Dario Guidotti, Stefano Demarchi",
    author_email="dguidotti@uniss.it, stefano.demarchi@edu.unige.it",
    license='GNU General Public License with Commons Clause License Condition v1.0',
    description="Package for the design, training, pruning and verification of neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeVerTools/pyNeVer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=['numpy', 'onnx', 'torch', 'torchvision', 'ortools', 'pysmt', 'multipledispatch'],
)
