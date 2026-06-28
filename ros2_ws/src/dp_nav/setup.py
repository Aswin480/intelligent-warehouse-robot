import os
from glob import glob
from setuptools import setup

package_name = "dp_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aditya K",
    license="MIT",
    entry_points={
        "console_scripts": [
            "planner_node = dp_nav.planner_node:main",
        ],
    },
)
