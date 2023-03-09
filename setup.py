import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyNeVer",
    version="0.0.2.alpha5",
    author="Dario Guidotti",
    author_email="dario.guidotti@edu.unige.it",
    license='GNU General Public License with Commons Clause License Condition v1.0',
    description="Package for the training, pruning and verification of neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeVerTools/pyNeVer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy', 'scipy', 'ortools', 'onnx', 'torch', 'torchvision', 'pysmt', 
        "tensorflow;platform_system=='Linux'",
        "tensorflow;platform_system=='Windows'"],
)

