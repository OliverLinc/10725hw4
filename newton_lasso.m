function [beta,xi,obj_lr_v] = ...
newton_lasso(beta, xi, X, y, lambda, t, step_alpha, step_beta, TOL)
%NEWTON_LASSO Run Newton update for given t in Barrier method
%   Parameters are consistent with function BARRIER_LASSO
shape = size(X);
n = shape(1);
p = shape(2)-1;

flag = TOL;

obj = obj_lasso_lr(X,y,beta,lambda);
obj_lr_v = [obj];
while (flag >= TOL)
    
    beta_0 = beta(1);
    beta_n = beta(2:end);

    mu = 1./(1+exp(-X*beta));
    D = diag(mu.*(1-mu));
    g = X'*(mu-y);
    H = X' * D * X;
    
    g_F = t * [g; lambda*ones(p, 1)];
    g_F = g_F + ...
        [0; 1./(xi-beta_n)-1./(xi+beta_n); -1./(xi-beta_n)-1./(xi+beta_n)];

    H_F = t * [H, zeros(p+1,p);zeros(p,2*p+1)];
    temp_diag = 1./(xi-beta_n).^2 + 1./(xi+beta_n).^2;
    H_F = H_F + diag([0; temp_diag; temp_diag]);
    % backtrack
    step_t = backtrack_step(X,y,xi, beta, H_F, g_F, ...
        p,t,lambda,step_alpha,step_beta);
    
    % update
    x_next = [beta;xi] - step_t*inv(H_F)*g_F;
    beta = x_next(1:p+1);
    xi = x_next(p+2:end);

    obj_new = obj_lasso_lr(X,y,beta,lambda);
    flag = abs(obj_new-obj);
    obj = obj_new;
    obj_lr_v = [obj_lr_v, obj];
    
end

end

