
% This file contains parameters and calculations needed for running
% MatLab with rotorS ROS package for interfacing with a position controlled
% drone

%%
clc;
close all;
clear all;
%%

% Define constants
FIXED_STEP_SIZE =   0.01;
POS_SUB_DT =        0.01;
ATTI_SUB_DT =       0.01;
IMU_SUB_DT =        0.01;
ODOMETRY_SUB_DT =   0.01;
GAZEBO_SYNC_DT =    0.01;
RAD2DEG =           180/pi;
DEG2RAD =           pi/180;
THRUST_OFFSET =     15;

% define a counter
COUNTER = 1;

% ex6.8
% create the map
maze_1
% plan a route via A*
start = [0,0]+1;
end_ = [3,6]+1;
route = astar_2d(map, start, end_, 1);
route = route-1;
route = [route,ones(size(route, 1),1)];

% ex6.9
% create the map
maze_1_3D
% plan a route via A*
start = [1,1,1];
end_ = [9,10,3];
route = astar_3d(map, start, end_, 1);
route(:,1:2) = route(:,1:2)-1;
route(:,3) = route(:,3)*2-1;

% ex6.10
THRUST_OFFSET = 7.025;