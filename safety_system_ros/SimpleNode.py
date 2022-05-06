import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive


class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')

        goal_topic = '/goal_pose'
        self.goal_sub = self.create_subscription(
            PoseStamped,
            goal_topic,
            self.goal_callback,
            10)

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(1.0, self.timer_callback)


    def goal_callback(self, msg):
        self.get_logger().info('Received goal: %s' % msg)

    def timer_callback(self):
        self.get_logger().info('Hello, world! --> Publishing time')

        cmd_msg = Twist()
        cmd_msg.linear.x = 0.5
        cmd_msg.angular.z = 0.0
        self.cmd_publisher.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


