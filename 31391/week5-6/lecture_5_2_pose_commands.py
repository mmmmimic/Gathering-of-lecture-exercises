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
import math
 
from std_msgs.msg import String
 
def move_group_python_interface_tutorial():
  ## BEGIN_TUTORIAL
  ## First initialize moveit_commander and rospy.
  print "============ Starting tutorial setup"
  moveit_commander.roscpp_initialize(sys.argv)
  rospy.init_node('move_group_python_interface_tutorial',
                  anonymous=True)
 
  robot = moveit_commander.RobotCommander()
  scene = moveit_commander.PlanningSceneInterface()
  group = moveit_commander.MoveGroupCommander("Arm")
 
  ## trajectories for RVIZ to visualize.
  display_trajectory_publisher = rospy.Publisher(
                                      '/move_group/display_planned_path',
                                      moveit_msgs.msg.DisplayTrajectory)
 
  print "============ Starting tutorial "
  ## We can get the name of the reference frame for this robot
  print "============ Reference frame: %s" % group.get_planning_frame()
  ## We can also print the name of the end-effector link for this group
  print "============ End effector frame: %s" % group.get_end_effector_link()
  ## We can get a list of all the groups in the robot
  print "============ Robot Groups:"
  print robot.get_group_names()
  ## Sometimes for debugging it is useful to print the entire state of the
  ## robot.
  print "============ Printing robot state"
  print robot.get_current_state()
  print "============"
 
  ## Let's setup the planner
  #group.set_planning_time(0.0)
  group.set_goal_orientation_tolerance(0.01)
  group.set_goal_tolerance(0.01)
  group.set_goal_joint_tolerance(0.01)
  group.set_num_planning_attempts(100)
 
  ## Planning to a Pose goal
  print "============ Generating plan 1"
 
  pose_goal = group.get_current_pose().pose
  pose_goal.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0., -math.pi/2, 0.))
  pose_goal.position.x =0.40
  pose_goal.position.y =-0.10
  pose_goal.position.z =1.35
  print pose_goal
  group.set_pose_target(pose_goal)
 
 
  ## Now, we call the planner to compute the plan
  plan1 = group.plan()
 
  print "============ Waiting while RVIZ displays plan1..."
  rospy.sleep(0.5)
 
 
  ## You can ask RVIZ to visualize a plan (aka trajectory) for you.
  print "============ Visualizing plan1"
  display_trajectory = moveit_msgs.msg.DisplayTrajectory()
  display_trajectory.trajectory_start = robot.get_current_state()
  display_trajectory.trajectory.append(plan1)
  display_trajectory_publisher.publish(display_trajectory);
  print "============ Waiting while plan1 is visualized (again)..."
  rospy.sleep(2.)
 
  #If we're coming from another script we might want to remove the objects
  if "table" in scene.get_known_object_names():
    scene.remove_world_object("table")
  if "table2" in scene.get_known_object_names():
    scene.remove_world_object("table2")
  if "groundplane" in scene.get_known_object_names():
    scene.remove_world_object("groundplane")
 
  ## Moving to a pose goal
  group.go(wait=True)
 
  ## second movement
  pose_goal = group.get_current_pose().pose
  pose_goal.position.x =0.40
  pose_goal.position.y =-0.10
  pose_goal.position.z =1.35
  pose_goal.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0.,  0.  , 0.))
  group.set_pose_target(pose_goal)
 
  plan1 = group.plan()
  rospy.sleep(2.)
 
 ## You can ask RVIZ to visualize a plan (aka trajectory) for you.
  display_trajectory = moveit_msgs.msg.DisplayTrajectory()
  display_trajectory.trajectory_start = robot.get_current_state()
  display_trajectory.trajectory.append(plan1)
  display_trajectory_publisher.publish(display_trajectory);
 
  rospy.sleep(2)
 
  group.go(wait=True)
  rospy.sleep(2.)
 
  ## When finished shut down moveit_commander.
  moveit_commander.roscpp_shutdown()
 
  ## END_TUTORIAL
  print "============ STOPPING"
  R = rospy.Rate(10)
  while not rospy.is_shutdown():
    R.sleep()
if __name__=='__main__':
  try:
    move_group_python_interface_tutorial()
  except rospy.ROSInterruptException:
    pass
