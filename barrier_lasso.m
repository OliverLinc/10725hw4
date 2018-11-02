function [beta,xi,obj,obj_list] = ...
    barrier_lasso(t,u,p,X,y,lambda,step_alpha,step_beta,TOL,TOL_barrier)
%BARRIER_LASSO Run Barrier method for lasso
%   Start with t0, run Newton update to get optimal beta,
%   then increase t with parameter u.
%   t: initial barrier parameter
%   u: Barrier update
%   p: number of features (without intersect term)
%   X, y: training data
%   lambda: LASSO penalty
%   step_alpha, step_beta: backtrack parameter
%   TOL: Newton update tol
%   TOL_barrier: Barrier method tol
[beta,xi] = find_strict_feasible(p);
m = 2*p;
obj_list = [];
while 1
    [beta,xi,obj_l]=...
        newton_lasso(beta,xi,X,y,lambda,t,step_alpha,step_beta,TOL);
    obj_list = [obj_list, obj_l];
    if (m/t <= TOL_barrier)
        break;
    end
    obj = obj_lasso_lr(X,y,beta,lambda);
    t = u * t;
end
end

