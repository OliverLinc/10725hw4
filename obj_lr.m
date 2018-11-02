function [v] = obj_lr(X,y,beta)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here
v = -sum(y.*(X*beta))+sum(log(1+exp(X*beta)));
end

