#!/usr/bin/env python
import rospy
import numpy as np

from std_msgs.msg import String
from turtlesim import msg

def sub_cal(msg):
    rospy.loginfo("position=(%f, %f), direction=%f", msg.x, msg.y, msg.theta)
    
rospy.Subscriber('turtle1/pose', msg.Pose, sub_cal, queue_size = 1000)

rospy.init_node('cmd_vel_listener')
rospy.spin()
rospy.loginfo()
