#!/usr/bin/env python
# BEGIN ALL
import rospy
from sensor_msgs.msg import LaserScan
 
# BEGIN MEASUREMENT
def scan_callback(msg):
  range_ahead = msg.ranges[0]
  tmp=[msg.ranges[0]]
  for i in range(1,21):
    tmp.append(msg.ranges[i])
  for i in range(len(msg.ranges)-21,len(msg.ranges)):
    tmp.append(msg.ranges[i])
  print "range ahead: %0.1f" % min(tmp)
  # END MEASUREMENT
 
rospy.init_node('range_ahead')
scan_sub = rospy.Subscriber('scan', LaserScan, scan_callback)
rospy.spin()
# END ALL