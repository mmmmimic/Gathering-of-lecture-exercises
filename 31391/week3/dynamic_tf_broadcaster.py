#!/usr/bin/env python 
import roslib
roslib.load_manifest('hello_ros')

import rospy
import tf
from math import *

if __name__=="__main__":
	rospy.init_node("dynamic_tf_broadcaster")
	br=tf.TransformBroadcaster()
	rate=rospy.Rate(10)
	while not rospy.is_shutdown():
		t=rospy.Time.now().to_sec()*pi
		br.sendTransform((2.0*sin(t),2.0*cos(t),0.0),(0.0,0.0,0.0,1.0),rospy.Time.now(),"carrot1", "turtle1")
		rate.sleep()
		