function [y] = predict_y(X, beta)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
shape = size(X);
n = shape(1);
p = shape(2);
if (p ~= length(beta))
    X = [ones(n,1), X];
end

y = 1./(1+exp(-X*beta));
y(y>=0.5) = 1;
y(y<0.5) = 0;
end

