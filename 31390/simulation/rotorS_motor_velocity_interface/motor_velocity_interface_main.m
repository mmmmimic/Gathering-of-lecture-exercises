
% This file contains parameters and calculations needed for running
% MatLab with rotorS ROS package for interfacing with an attitude controlled
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
THRUST_OFFSET =     7.024;
MAX_ROTOR_VEL =     838;
Z_OFFSET = 0.06;

% PARAMETERS OF THE DRONE
MASS = 0.716;
L = 0.17;
ROTOR_RAD = 0.1;
IXX = 0.007;
IYY = 0.007;
IZZ = 0.012;
GRAVITY = 9.81;
K = 8.54858e-6;
B = 1.3678e-7;
