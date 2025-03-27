import json
import time
import numpy as np
from pydobot import Dobot

port = "/dev/tty.usbserial-130"

device = Dobot(port=port)
pose = device._get_pose()

device.speed(350, 370)
print(device.pose())
device.move_to(120, 0,0 , 0)