#!/usr/bin/env python
 
import rospy
import actionlib
 
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
 
 
waypoints = [  
    [(1.13, -1.6, 0.0), (0.0, 0.0, -0.16547, -0.986213798314)],
    [(0.13, 1.93, 0.0), (0.0, 0.0, -0.64003024, -0.76812292098)]
]
 
 
def goal_pose(pose):  
    goal_pose = MoveBaseGoal()
    goal_pose.target_pose.header.frame_id = 'map'
    goal_pose.target_pose.pose.position.x = pose[0][0]
    goal_pose.target_pose.pose.position.y = pose[0][1]
    goal_pose.target_pose.pose.position.z = pose[0][2]
    goal_pose.target_pose.pose.orientation.x = pose[1][0]
    goal_pose.target_pose.pose.orientation.y = pose[1][1]
    goal_pose.target_pose.pose.orientation.z = pose[1][2]
    goal_pose.target_pose.pose.orientation.w = pose[1][3]
 
    return goal_pose
 
 
if __name__ == '__main__':
    rospy.init_node('patrol')
 
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction) 
    client.wait_for_server()
   
    while True:
        for pose in waypoints:   
            goal = goal_pose(pose)
            client.send_goal(goal)
            client.wait_for_result()
            rospy.sleep(3)