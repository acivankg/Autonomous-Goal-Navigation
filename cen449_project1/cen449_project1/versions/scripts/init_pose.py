#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Node('init_pose_node')

    pub = node.create_publisher(
        PoseWithCovarianceStamped,
        '/initialpose',
        10
    )

    node.get_logger().info('Publishing initial pose to /initialpose ...')

    msg = PoseWithCovarianceStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'map'   # AMCL map frame'inde bekler


    x = 0.0
    y = 0.0
    yaw_deg = 0.0          

    yaw = math.radians(yaw_deg)
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0

    msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.0)


    msg.pose.covariance[0] = 0.25   # x
    msg.pose.covariance[7] = 0.25   # y
    msg.pose.covariance[35] = 0.068 # yaw


    for _ in range(5):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info('Initial pose published. Exiting.')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
