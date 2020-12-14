clear; close all; clc;
%% 
tau = 0.01;
A = genA(tau);
cond_ = cond(A);
eig_ = eigs(A, 1, 'sm');
det_ = det(A);
disp('tau: '+string(tau));
disp('cond: '+string(cond_));
disp('smallest eigen value: '+string(eig_));
disp('determinant: '+string(det_));

%%
tau = 0.05;
A = genA(tau);
cond_ = cond(A);
eig_ = eigs(A, 1, 'sm');
det_ = det(A);
disp('tau: '+string(tau));
disp('cond: '+string(cond_));
disp('smallest eigen value: '+string(eig_));
disp('determinant: '+string(det_));

%%
tau = 0.09;
A = genA(tau);
cond_ = cond(A);
eig_ = eigs(A, 1, 'sm');
det_ = det(A);
disp('tau: '+string(tau));
disp('cond: '+string(cond_));
disp('smallest eigen value: '+string(eig_));
disp('determinant: '+string(det_));

%%
tau = 0.3;
A = genA(tau);
cond_ = cond(A);
eig_ = eigs(A, 1, 'sm');
det_ = det(A);
disp('tau: '+string(tau));
disp('cond: '+string(cond_));
disp('smallest eigen value: '+string(eig_));
disp('determinant: '+string(det_));