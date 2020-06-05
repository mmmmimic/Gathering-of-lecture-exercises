#!/usr/bin/env python
import sys
import rospy
from turtlesim.srv import *

def call_service(x, theta):
	rospy.wait_for_service("/turtle1/teleport_relative")
	try:
		teleport_relative=rospy.ServiceProxy('/turtle1/teleport_relative', TeleportRelative)
		resp1=teleport_relative(x,theta)
		return True
	except rospy.ServiceException,e:
		print "Service call failed: %s"%e
		return False

def usage():
	return "%s[x,theta]"%sys.argv[0]

if __name__=="__main__":
	if len(sys.argv)==3:
		x=float(sys.argv[1])
		theta=float(sys.argv[2])
	else:
		print usage()
		sys.exit(1)
	print "Requesting a relative teleport linear:%s, angular:%s"%(x,theta)
	print "The execution is %s"%call_service(x, theta)
	

