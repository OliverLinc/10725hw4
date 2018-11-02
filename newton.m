function [beta,obj] = newton(X, y, step_alpha, step_beta, TOL, f_star)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

shape = size(X);
n = shape(1);
p = shape(2);

beta = zeros(p, 1);

flag = TOL;

obj = obj_lr(X,y,beta);
obj_list = [];
while (flag >= TOL)
    obj_list = [obj_list, obj];
    mu = 1./(1+exp(-X*beta));
    D = diag(mu.*(1-mu));
    g = X'*(mu-y);
    H = X' * D * X;
    % backtrack
    t = 1;
    H_inv = inv(H);
    step_v = -H_inv*g;
    while (obj_lr(X, y, beta+t*step_v)>obj_lr(X,y,beta)+step_alpha*t*g'*step_v)
        t = step_beta * t;
    end
    beta = beta - t*H_inv*g;
    obj_new = obj_lr(X,y,beta);
    flag = abs(obj_new-obj);
    obj = obj_new;
end

semilogy(1:length(obj_list), obj_list-f_star);
xlabel('Iteration');
ylabel('$\log(f-f^\star)$', 'Interpreter', 'latex');
title('Objective function vs iteration');
end