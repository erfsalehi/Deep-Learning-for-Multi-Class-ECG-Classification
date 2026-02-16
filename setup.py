from setuptools import setup, find_packages

setup(
    name="cardiovascular_ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "tensorflow",
        "wfdb",
        "neurokit2",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
    ],
)
