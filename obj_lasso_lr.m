function [v] = obj_lasso_lr(X,y,beta,lambda)
%UNTITLED17 Summary of this function goes here
%   Detailed explanation goes here
beta_n = beta(2:end);
v = obj_lr(X,y,beta) + lambda*sum(abs(beta_n));
end

