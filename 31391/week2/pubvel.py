#!/usr/bin/env python
import rospy
import numpy as np

from std_msgs.msg import String
from geometry_msgs.msg import Twist

#initialize publisher
p=rospy.Publisher('turtle1/cmd_vel', Twist, queue_size=1000)

#initialize node
rospy.init_node('publish_velocity')
r = rospy.Rate(2)

while not rospy.is_shutdown():
    t = Twist()
    t.angular.z = 2*np.random.rand()-1
    t.linear.x = np.random.rand()
    
    rospy.loginfo("Ang.z is %s"%t.angular.z)
    rospy.loginfo("Lin.x is %s"%t.linear.x)
    
    p.publish(t)
    r.sleep()
    
