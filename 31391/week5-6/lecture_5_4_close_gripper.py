#!/usr/bin/env python
import roslib
roslib.load_manifest('hello_ros')
 
import sys
import copy
import rospy
import tf_conversions
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import shape_msgs.msg as shape_msgs
from sensor_msgs.msg import JointState
from numpy import zeros, array, linspace
from math import ceil
 
currentJointState = JointState()
def jointStatesCallback(msg):
  global currentJointState
  currentJointState = msg
if __name__ == '__main__':
  rospy.init_node('test_publish')
 
  # Setup subscriber
  #rospy.Subscriber("/joint_states", JointState, jointStatesCallback)
 
  pub = rospy.Publisher("/jaco/joint_control", JointState, queue_size=1)
 
  currentJointState = rospy.wait_for_message("/joint_states",JointState)
  print 'Received!'
  currentJointState.header.stamp = rospy.get_rostime()
  tmp = 0.7
  #tmp_tuple=tuple([tmp] + list(currentJointState.position[1:]))
  currentJointState.position = tuple(list(currentJointState.position[:6]) + [tmp] + [tmp]+ [tmp])
  rate = rospy.Rate(10) # 10hz
  for i in range(3):
    pub.publish(currentJointState)
    print 'Published!'
    rate.sleep()
 
  print 'end!'
