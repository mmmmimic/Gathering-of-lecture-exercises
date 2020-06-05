#!/usr/bin/env python 
import roslib
roslib.load_manifest('hello_ros')

import rospy
import tf

if __name__=="__main__":
	rospy.init_node("fixed_tf_broadcaster")
	br=tf.TransformBroadcaster()
	rate=rospy.Rate(10)
	while not rospy.is_shutdown():
		br.sendTransform((0.0,2.0,0.0),(0.0,0.0,0.0,1.0),rospy.Time.now(),"carrot1", "turtle1")
		rate.sleep()
		