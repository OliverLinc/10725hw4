function [step_t] = ...
    backtrack_step(X,y,xi,beta,H,g,p,t,lambda,step_alpha,step_beta)
%BACKTRACK_STEP Compute backtrack step size for Newton update in barrier
%   Parameters are consistent with BARRIER_LASSO

beta_n = beta(2:end);

direction = inv(H)*g;

d_beta_n = direction(2:p+1);
d_xi = direction(p+2:end);

% ensure xi-beta_n>0
diff_d = d_xi - d_beta_n;
diff = xi - beta_n;
diff_d(diff_d<0) = diff(diff_d<0);
step_t = min(1, min(diff./diff_d));

% ensure xi+beta_n>0
diff_d_sum = d_xi + d_beta_n;
diff_sum = xi + beta_n;
diff_d_sum(diff_d_sum<0) = diff_sum(diff_d_sum<0);
step_t = 0.99*min(step_t, min(diff_sum./diff_d_sum));

v = -direction;

while 1 
    temp = [beta;xi]+step_t*v;
    temp_beta = temp(1:p+1);
    temp_xi = temp(p+2:end);
    if (obj_lasso_barrier(X,y,temp_beta,temp_xi,lambda,t)<= ...
            obj_lasso_barrier(X,y,beta,xi,lambda,t)+...
            step_alpha*step_t*g'*v)
       break;
    end
    step_t = step_t * step_beta;
end
