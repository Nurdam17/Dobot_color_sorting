from rclpy.node import Node
from std_msgs.msg import String
from pydobotplus import Dobot
import numpy as np
import json
import time
from threading import Event
from collections import deque


class DobotExecutorNode(Node):
    def __init__(self):
        super().__init__('dobot_executor_node')
        self.subscription = self.create_subscription(
            String,
            'dobot_command',
            self.listener_callback,
            10
        )

        self.processing_event = Event()
        self.processing_event.clear()

        # Последние обработанные позиции (ограничено 10)
        self.handled_positions = deque(maxlen=10)
        self.last_grab_time = 0

        self.device = Dobot(port='/dev/ttyUSB0')
        self.device.speed(450, 500)

        self.R_m = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ])
        self.P_robot_to_camera = np.array([200, 0, 280])
        self.diff_x = 90
        self.diff_y = -130

    def is_already_handled(self, x, y, z, threshold=15.0):
        x, y, z = round(x, 1), round(y, 1), round(z, 1)
        for pos in self.handled_positions:
            dx = abs(x - pos[0])
            dy = abs(y - pos[1])
            dz = abs(z - pos[2])
            if dx < threshold and dy < threshold and dz < threshold:
                return True
        return False

    def listener_callback(self, msg: String):
        if self.processing_event.is_set():
            self.get_logger().warn("Already processing. Skipping new message.")
            return

        try:
            obj = json.loads(msg.data)
            model_name = obj["model_name"]
            x = round(obj["X_robot_mm"], 1)
            y = round(obj["Y_robot_mm"], 1)
            z = round(obj["Z_robot_mm"], 1)

            if self.is_already_handled(x, y, z):
                self.get_logger().info("Already handled this position. Ignoring.")
                return

            if time.time() - self.last_grab_time < 5:
                self.get_logger().info("Too soon to grab again.")
                return

            self.processing_event.set()
            self.handled_positions.append((x, y, z))
            self.last_grab_time = time.time()

            P_camera = np.array([x, y, z])
            P_robot = self.P_robot_to_camera + np.dot(self.R_m, P_camera)
            P_robot[1] = -P_robot[1]

            self.get_logger().info(f"[{model_name}] Move to X={P_robot[0]:.1f}, Y={P_robot[1]:.1f}, Z={50:.1f}")

            self.device.move_to(x=P_robot[0] + self.diff_x, y=P_robot[1] + self.diff_y, z=50)
            self.device.move_to(x=P_robot[0] + self.diff_x, y=P_robot[1] + self.diff_y, z=-60)

            if model_name in ["yellow", "red"]:
                self.device.suck(True)

            time.sleep(0.1)
            self.device.move_to(x=P_robot[0] + self.diff_x, y=P_robot[1] + self.diff_y, z=50)

            if model_name == "yellow":
                self.device.move_to(x=110, y=250, z=50)
                self.device.move_to(x=110, y=250, z=35)
                self.device.suck(False)
                self.device.conveyor_belt_distance(speed_mm_per_sec=100, distance_mm=100, direction=-1)
            elif model_name == "red":
                self.device.move_to(x=110, y=300, z=50)
                self.device.move_to(x=110, y=300, z=35)
                self.device.suck(False)
                self.device.conveyor_belt_distance(speed_mm_per_sec=100, distance_mm=100, direction=-1)
            else:
                self.device.move_to(x=120, y=0, z=0)

            self.device.move_to(x=120, y=0, z=0)
            time.sleep(0.1)

        except Exception as e:
            self.get_logger().error(f"Error while processing object: {e}")

        finally:
            self.processing_event.clear()


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = DobotExecutorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
