#!/usr/bin/env python3
"""
Circle-test logger for friction calibration (writes t, v, yaw_rate, steer to CSV).

Drive a constant steering angle with the joystick (or a fixed /drive command) and raise the speed
slowly until the car slides. Then run on the laptop:
    python tools/calibrate_mu.py circle_test.csv --plot circle_test.png

Yaw rate MUST come from a sensor that sees slip: the IMU gyro (default /sensors/imu/raw) or the
particle-filter pose (yaw_source:=pose -> differentiated /pf/pose/odom heading). Do NOT use the
/odom twist from vesc_to_odom: it computes yaw rate from the steering command, so it never slips.
UNTESTED ON ROS (the analysis side is tested in the simulator).
"""
import math
import os

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class CircleLogger(Node):
    def __init__(self):
        super().__init__('circle_test_logger')
        for k, v in {'speed_topic': '/odom', 'yaw_source': 'imu', 'imu_topic': '/sensors/imu/raw',
                     'pose_topic': '/pf/pose/odom', 'drive_topic': '/drive',
                     'out_csv': os.path.expanduser('~/circle_test.csv'), 'rate_hz': 50.0}.items():
            self.declare_parameter(k, v)
            setattr(self, k, self.get_parameter(k).value)
        self.v = float('nan')
        self.r = float('nan')
        self.steer = float('nan')
        self._last_pose = None
        self.create_subscription(Odometry, self.speed_topic, self.on_speed, 20)
        if self.yaw_source == 'imu':
            self.create_subscription(Imu, self.imu_topic, self.on_imu, 50)
        else:
            self.create_subscription(Odometry, self.pose_topic, self.on_pose, 50)
        self.create_subscription(AckermannDriveStamped, self.drive_topic, self.on_drive, 20)
        self.f = open(self.out_csv, 'w')
        self.f.write('t,v,yaw_rate,steer\n')
        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info('logging to %s (yaw rate from %s)' % (self.out_csv, self.yaw_source))

    def on_speed(self, msg):
        self.v = abs(msg.twist.twist.linear.x)

    def on_imu(self, msg):
        self.r = msg.angular_velocity.z

    def on_pose(self, msg):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        if self._last_pose is not None:
            t0, y0 = self._last_pose
            if t - t0 > 1e-3:
                d = (yaw - y0 + math.pi) % (2 * math.pi) - math.pi
                self.r = d / (t - t0)
        self._last_pose = (t, yaw)

    def on_drive(self, msg):
        self.steer = msg.drive.steering_angle

    def tick(self):
        if math.isnan(self.v) or math.isnan(self.r):
            return
        t = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        s = 0.0 if math.isnan(self.steer) else self.steer
        self.f.write('%.4f,%.4f,%.5f,%.5f\n' % (t, self.v, self.r, s))

    def destroy_node(self):
        self.f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CircleLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
