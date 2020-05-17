function [ ] = resetOdoCov(socket)
%RESETODOCOV Summary of this function goes here
%   Detailed explanation goes here
    mssendraw(socket,uint8(['resetodocov' 10]));
end
