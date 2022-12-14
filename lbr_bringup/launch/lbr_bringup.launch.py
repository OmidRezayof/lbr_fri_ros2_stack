from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.actions.declare_launch_argument import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch.substitutions.launch_configuration import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    # Evaluate frequently used variables
    model = LaunchConfiguration("model").perform(context)

    # Load robot description
    robot_description_content = Command(
        [
            FindExecutable(name="xacro"), " ",
            PathJoinSubstitution(
                [FindPackageShare("lbr_description"), "urdf/{}/{}.urdf.xacro".format(model, model)]
            ), " ",
            "robot_name:=", LaunchConfiguration("robot_name"), " ",
            "sim:=", LaunchConfiguration("sim")
        ]
    )

    # Load controls
    control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("lbr_bringup"),
                "launch",
                "lbr_control.launch.py"
            ])
        ), launch_arguments=[
            ("robot_description", robot_description_content),
            ("controller_configurations_package", LaunchConfiguration("controller_configurations_package")),
            ("controller_configurations", LaunchConfiguration("controller_configurations")),
            ("controller", LaunchConfiguration("controller")),
            ("sim", LaunchConfiguration("sim"))
        ]
    )

    # Gazebo simulation 
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("lbr_bringup"),
                "launch",
                "lbr_simulation.launch.py"
            ])
        ), 
        launch_arguments=[
           ("robot_name", LaunchConfiguration("robot_name"))
        ],
        condition=IfCondition(LaunchConfiguration("sim"))
    )

    # Rviz
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        parameters=[
            {"robot_description": robot_description_content}
        ],
        arguments=["-d", PathJoinSubstitution(
            [FindPackageShare("lbr_description"), "config/config.rviz"]
        )]
    )

    return [
        simulation,
        control,
        rviz2
    ]


def generate_launch_description():
    model_arg = DeclareLaunchArgument(
        name="model",
        default_value="iiwa7",
        description="Desired LBR model. Use model:=iiwa7/iiwa14/med7/med14."
    )

    robot_name_arg = DeclareLaunchArgument(
        name="robot_name",
        default_value="lbr",
        description="Set robot name."
    )

    sim_arg = DeclareLaunchArgument(
        name="sim",
        default_value="true",
        description="Launch robot in simulation or on real setup."
    )

    controller_configurations_package_arg = DeclareLaunchArgument(
        name="controller_configurations_package",
        default_value="lbr_bringup",
        description="Package that contains controller configurations."
    )

    controller_configurations_arg = DeclareLaunchArgument(
        name="controller_configurations",
        default_value="config/lbr_controllers.yml",
        description=
            "Relative path to controller configurations YAML file.\n"
            "\tNote that the joints in the controllers must be named according to the robot_name."
    )

    controller_arg = DeclareLaunchArgument(
        name="controller",
        default_value="position_trajectory_controller",
        description="Robot controller."
    )

    return LaunchDescription([
        model_arg,
        robot_name_arg,
        sim_arg,
        controller_configurations_package_arg,
        controller_configurations_arg,
        controller_arg,
        OpaqueFunction(function=launch_setup)
    ])
