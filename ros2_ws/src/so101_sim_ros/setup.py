from glob import glob

from setuptools import setup

package_name = "so101_sim_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "sim_node = so101_sim_ros.sim_node:main",
            "teleop_udp_node = so101_sim_ros.teleop_udp_node:main",
            "policy_node = so101_sim_ros.policy_node:main",
        ],
    },
)
