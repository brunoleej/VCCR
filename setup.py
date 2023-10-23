from distutils.core import setup
from setuptools import find_packages

setup(
    name="vccr",
    packages=find_packages(),
    version="0.0.1",
    description="Visual Navigation Urgent Circumstances via Counterfactual Reasoning in CARLA Simulator",
    long_description=open("./README.md").read(),
    author="Bruno lee",
    author_email="brunoleej@gmail.com",
    url="https://brunoleej.github.io/portfolio/portfolio-3/",
    entry_points={
        "console_scripts": (
            "vccr=vccr.train_agent:main",
            "viskit.scripts.console_scripts:main"
        )
    },
    requires=(),
    zip_safe=True,
    license="MIT",
)
