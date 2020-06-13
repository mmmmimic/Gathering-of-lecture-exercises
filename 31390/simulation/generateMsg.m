examplePackages = fullfile(fileparts(which('rosgenmsg')), 'customMessages')
userFolder = '/home/ananda/ROS/customMessages/'
copyfile(userFolder, examplePackages)
folderpath = userFolder;
rosgenmsg(examplePackages)

%%

addpath('/home/ananda/Documents/MATLAB/SupportPackages/R2019b/toolbox/ros/supportpackages/roscustommsg/customMessages/matlab_gen/msggen')
%savepath