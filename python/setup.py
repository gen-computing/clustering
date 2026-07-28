from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import subprocess
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + os.sys.executable,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_PYTHON=ON",
        ]

        build_args = ["--config", "Release", "-j4"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

setup(
    name="clustering",
    version="1.0.0",
    author="Clustering Engine",
    description="High-performance C++ clustering engine with Python bindings",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("clustering._clustering")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=["numpy"],
)
