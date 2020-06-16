%========================================================
% @author: mxl
% @date: 06/15/2020
%--------------------------------------------------------
%%
clear; close all; clc;
addpath('/Gathering-of-lectures-and-exercises/31390/part4/Astar/');
%%
%// draw the graph
s = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, ...
    6, 7, 7, 8, 8, 9, 9, 10, 11, 11, 12, 12, 14, ...
    14, 15, 15, 16, 17, 17, 17, 18, 19, 19, 20, 22, 23, 23]+1;
t = [1, 2, 3, 4, 2, 5, 6, 7, 8, 5, 9, ...
    10, 6, 11, 7, 11, 12, 13, 13, 14, 15, 16, 23, 15, ...
    18, 19, 20, 21, 12, 19, 23, 17, 20, 23, 22, 21, 21, 22]+1;
G = graph(s, t);
figure;
plot(G);
%%
%// compress the map in a matrix
M = compress(s, t);
%% Exercise 4.1
%// Depth First Search
% start by stacking the lowest of the s numbers
% start from s0
start_node = 0;
end_node = 23;
[route, logs] = dfs(start_node, end_node, M);
path_len = length(route)-1;
%% Exercise 4.2
%// Breadth First Search
% start by stacking the lowest of the s numbers
% start from s0
start_node = 0;
end_node = 23;
[route, logs] = bfs(start_node, end_node, M);
path_len = length(route)-1;
%%
% from quora
% Advantages of BFS:-
%
% 1. Solution will definitely found out by BFS If there are some solution.
% 2. BFS will never get trapped in blind alley , means unwanted nodes.
% 3. If there are more than one solution then it will find solution with minimal steps.
%
% Disadvantages Of BFS :-
% 1. Memory Constraints As it stores all the nodes of present level to go for next level.
% 2. If solution is far away then it consumes time.
%
% Advantages Of DFS :-
% 1. Memory requirement is Linear WRT Nodes.
% 2. Less time and space complexity rather than BFS.
% 3. Solution can be found out by without much more search.
%
% Disadvantage of DFS :-
% 1. Not Guaranteed that it will give you solution.
% 2. Cut-off depth is smaller so time complexity is more.
% 3. Determination of depth until the search has proceeds.
%
%% Exercise 4.3
%// Dijkstra's algorithm
% generate Adjacency Matrix D
dist = [4.5, 5.4, 2.2, 2.2, 4.2, 3.6, 3.2, 2.2, 3.2, 1.4, 2.2, ...
    2, 1.4, 2.2, 2.2, 3.2, 6, 3, 4, 4.1, 6, 3.2, 5.1, 1.4, ...
    2, 3.2, 2.2, 4, 3.6, 4.1, 3.2, 3.6, 4.5, 2.8, 3.6, 3.6, 2, 3.2];
D = zeros(size(M, 1));
for i = 1:size(s, 2)
   D(s(i), t(i)) = dist(i); 
   D(t(i), s(i)) = dist(i); 
end
start_node = 0;
end_node = 23;
[route, logs] = Dijkstra(start_node, end_node, M, D);
%%
function [mtx] = compress(s, t)
% compress the graph into a matrix
assert(size(s, 2)==size(t, 2));
node_num = max(s);
mtx = zeros(node_num, node_num);
for i = 1:size(s, 2)
    mtx(s(i), t(i)) = 1;
    mtx(t(i), s(i)) = 1;
end
end