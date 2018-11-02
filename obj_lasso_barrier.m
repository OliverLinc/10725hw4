function [v] = obj_lasso_barrier(X,y,beta,xi,lambda, t)
%UNTITLED19 Summary of this function goes here
%   Detailed explanation goes here
v = t * (obj_lr(X,y,beta) + lambda*sum(xi));
beta_n = beta(2:end);
v = v-sum(log(xi-beta_n)+log(xi+beta_n));
end

