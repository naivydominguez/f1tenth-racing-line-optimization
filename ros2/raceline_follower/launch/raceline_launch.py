import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sh = get_package_share_directory('raceline_follower')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(sh, 'config', 'car_params.yaml')),
        DeclareLaunchArgument('waypoints', default_value=os.path.join(sh, 'waypoints', 'mintime_waypoints.csv')),
        DeclareLaunchArgument('track_csv', default_value=''),
        DeclareLaunchArgument('speed_scale', default_value='0.4'),
        Node(package='raceline_follower', executable='raceline_follower_node.py',
             name='raceline_follower', output='screen',
             parameters=[LaunchConfiguration('config'), {
                 'waypoints_path': LaunchConfiguration('waypoints'),
                 'track_csv': LaunchConfiguration('track_csv'),
                 'speed_scale': LaunchConfiguration('speed_scale'),
             }]),
    ])
