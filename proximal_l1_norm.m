function [prox] = proximal_l1_norm(beta, lambda, t)
%PROXIMAL_L1_NORM Returns the proximal of LASSO L1 penalty
%   beta_0 is unchanged. beta(2:end) is soft threshold with parameter 
%   lambda*t.
p = length(beta) - 1;
prox = zeros(p+1,1);
threshold = lambda * t;
prox(beta>threshold) = beta(beta>threshold) - threshold;
prox(beta<-threshold) = beta(beta<-threshold) + threshold;
prox(1) = beta(1);
end

