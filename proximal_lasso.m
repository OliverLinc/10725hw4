function [beta,obj_list] = proximal_lasso(X,y,lambda,step_beta,TOL)
%PROXIMAL_LASSO Run Proximal method for LASSO
%   X,y: training data
%   lambda: LASSO penalty
%   step_beta: backtrack update parameter
%   TOL: 1e-9
%   Returns beta* and objective values list during backtracking
shape = size(X);
n = shape(1);
p = shape(2)-1;
flag = TOL;
beta = zeros(p+1,1);
obj = obj_lasso_lr(X,y,beta,lambda);
obj_list = [obj];
while (flag >= TOL)

    mu = 1./(1+exp(-X*beta));
    D = diag(mu.*(1-mu));
    g = X'*(mu-y);
    H = X' * D * X;
    
    % backtracking
    t = 1;
    while 1
        beta_temp = beta - t * g;
        prox = proximal_l1_norm(beta_temp,lambda,t);
        Gt = (beta - prox)/t;
        obj_new = obj_lasso_lr(X,y,beta-t*Gt,lambda);
        obj_list = [obj_list, obj_new];
        if (obj_lr(X,y,beta-t*Gt)<=obj_lr(X,y,beta)-t*g'*Gt+t/2*norm(Gt)^2)
            break;
        end
        t = step_beta * t;
    end
    
    Gt = (beta - proximal_l1_norm(beta-t*g,lambda,t))/t;
    beta = beta - t*Gt;

    flag = abs(obj_new-obj);
    obj = obj_new;

end

