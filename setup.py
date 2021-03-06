#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# type: ignore

import os
import platform
import sys
import warnings

try:
    # Use setuptools if available, for install_requires (among other things).
    import setuptools
    from setuptools import setup
except ImportError:
    setuptools = None
    from distutils.core import setup


# The following code is copied from
# https://github.com/mongodb/mongo-python-driver/blob/master/setup.py
# to support installing without the extension on platforms where
# no compiler is available.
from distutils.command.build_ext import build_ext


class custom_build_ext(build_ext):
    """Allow C extension building to fail.
    The C extension speeds up websocket masking, but is not essential.
    """

    warning_message = """
********************************************************************
WARNING: %s could not
be compiled. No C extensions are essential for Tornado to run,
although they do result in significant speed improvements for
websockets.
%s
Here are some hints for popular operating systems:
If you are seeing this message on Linux you probably need to
install GCC and/or the Python development package for your
version of Python.
Debian and Ubuntu users should issue the following command:
    $ sudo apt-get install build-essential python-dev
RedHat and CentOS users should issue the following command:
    $ sudo yum install gcc python-devel
Fedora users should issue the following command:
    $ sudo dnf install gcc python-devel
MacOS users should run:
    $ xcode-select --install
********************************************************************
"""

    def run(self):
        try:
            build_ext.run(self)
        except Exception:
            e = sys.exc_info()[1]
            sys.stdout.write("%s\n" % str(e))
            warnings.warn(
                self.warning_message
                % (
                    "Extension modules",
                    "There was an issue with "
                    "your platform configuration"
                    " - see above.",
                )
            )

    def build_extension(self, ext):
        name = ext.name
        try:
            build_ext.build_extension(self, ext)
        except Exception:
            e = sys.exc_info()[1]
            sys.stdout.write("%s\n" % str(e))
            warnings.warn(
                self.warning_message
                % (
                    "The %s extension " "module" % (name,),
                    "The output above "
                    "this warning shows how "
                    "the compilation "
                    "failed.",
                )
            )


kwargs = {}

with open("docl/__init__.py") as f:
    ns = {}
    print(f.read(), ns)
    version = ns["version"]

with open("README.md") as f:
    kwargs["long_description"] = f.read()

if (
    platform.python_implementation() == "CPython"
    and os.environ.get("TORNADO_EXTENSION") != "0"
):
    # This extension builds and works on pypy as well, although pypy's jit
    # produces equivalent performance.
    #     kwargs["ext_modules"] = [
    #         Extension("tornado.speedups", sources=["tornado/speedups.c"])
    #     ]

    if os.environ.get("TORNADO_EXTENSION") != "1":
        # Unless the user has specified that the extension is mandatory,
        # fall back to the pure-python implementation on any build failure.
        kwargs["cmdclass"] = {"build_ext": custom_build_ext}


if setuptools is not None:
    python_requires = ">= 3.6"
    kwargs["python_requires"] = python_requires

setup(
    name="sleepace_verification_tool",
    version=version,
    packages=["slp_utils", "data-structure", "math", "science", "sort", "string"],
    package_data={
        # data files need to be listed both here (which determines what gets
        # installed) and in MANIFEST.in (which determines what gets included
        # in the sdist tarball)
        "sleepace_verification_tool": ["py.typed"],
    },
    author="chenbiao",
    author_email="chenbiao@sleepace.net",
    url="http://www.tornadoweb.org/",
    license="http://www.apache.org/licenses/LICENSE-2.0",
    description=(
        "Tornado is a Python web framework and asynchronous networking library,"
        " originally developed at FriendFeed."
    ),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    **kwargs
)
