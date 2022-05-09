import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class ResetPosition(Node):
    def __init__(self):
        super().__init__('reset_node')

        self.ego_reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose', 10)

        self.ego_reset()

    def ego_reset(self):
        msg = PoseWithCovarianceStamped() 

        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        self.ego_reset_pub.publish(msg)

        print("Finished publishing")


def main(args=None):
    rclpy.init(args=args)
    node = ResetPosition()
    rclpy.spin_once(node, timeout_sec=1)

if __name__ == '__main__':
    main()


