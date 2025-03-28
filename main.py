from serial.tools import list_ports
from pydobotplus import Dobot
import time
PORT_GP4 = 2
port = "/dev/tty.usbserial-130"

device = Dobot(port=port)
print(device.get_pose())
device.move_to(120, 0, 0,0)
device.close()