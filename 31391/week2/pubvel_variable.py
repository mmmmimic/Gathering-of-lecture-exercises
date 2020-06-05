#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Twist   #geometry_msgs/Twist
from hello_ros.msg import turtle


global speed_var
speed_var = 0.5

def callback(msg):
    rospy.loginfo("name=%s, speed=%f", msg.name, msg.speed)
    global speed_var
    speed_var = msg.speed
    
# initialize publisher
p = rospy.Publisher('turtle1/cmd_vel', Twist, queue_size=1000)
# initialize subscriber
rospy.Subscriber('turtle1/turtle_speed', turtle, callback, queue_size=1000)


#initialize node
rospy.init_node('publish_variable_velocity')
r = rospy.Rate(10) #Set Frequency


while not rospy.is_shutdown():
    
    t = Twist()  #initialize the message with zero
    # fill in the message
    t.angular.z = (2*np.random.rand()-1)*speed_var
    t.linear.x = (np.random.rand())*speed_var
    p.publish(t)
    r.sleep
    

