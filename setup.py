from setuptools import find_namespace_packages, setup


setup(
    name="salt-video",
    version="0.1.0",
    description="Self-consistent distribution matching for fast video generation",
    python_requires=">=3.10",
    packages=find_namespace_packages(
        include=["model*", "pipeline*", "trainer*", "utils*", "wan*", "demo_utils*"]
    ),
)
