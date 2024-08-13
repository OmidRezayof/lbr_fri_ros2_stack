from setuptools import setup

package_name = "lbr_demos_py"

setup(
    name=package_name,
    version="2.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mhubii",
    maintainer_email="m.huber_1994@hotmail.de",
    description="Python demos for lbr_ros2_control.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "joint_sine_overlay = lbr_demos_py.joint_sine_overlay:main",
            "joint_trajectory_client = lbr_demos_py.joint_trajectory_client:main",
            "torque_sine_overlay = lbr_demos_py.torque_sine_overlay:main",
            "wrench_sine_overlay = lbr_demos_py.wrench_sine_overlay:main",
            "pose_planning = lbr_demos_py.pose_planning:main",
            "move_2_cart = lbr_demos_py.move_2_cart:main",
            "move_to_pose_client = lbr_demos_py.move_to_pose_client:main",
            "long_lines = lbr_demos_py.long_lines:main",
            "submerged_demo = lbr_demos_py.submerged_demo:main",
            "Drill_Demo = lbr_demos_py.Drill_Demo:main",
            "axyb_calibration_ros2 = lbr_demos_py.axyb_calibration_ros2:main",
        ],
    },
)
