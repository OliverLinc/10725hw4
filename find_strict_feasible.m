function [beta, xi] = find_strict_feasible(p)
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here
% 
% f = [zeros(2*p+1,1); 1];
% I_m = eye(p);
% A = [I_m, -I_m; -I_m, -I_m];
% A = [zeros(2*p,1), A, -ones(2*p,1)];
% b = zeros(2*p,1);
% 
% x = linprog(f, A, b);
% beta = x(1:p+1);
% xi = x(p+2:2*p+1);

beta = zeros(p+1,1);
xi = ones(p,1);
end

