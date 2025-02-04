function [neglogp, d_logp, d2_logp]= GMRF_taylor(x_0, y, A, Q)
% GMRF_TAYLOR Taylor expansion of the conditional for non-Gaussian observations
%
% [logp, d_logp, d2_logp]= GMRF_taylor_Po(x_0, y, A, Q)
%
% x_0 = value at which to compute taylor expansion
% y = the data vector, as a column with n elements
% A = the observation matrix, sparse n-by-(N+Nbeta)
% Q = the precision matrix, sparse (N+Nbeta)-by-(N+Nbeta)
%
% Function should return value, gradient and Hessian of:
% -log(p(X|Y)) ~ -log(p(Y|X)) - log(p(X)) = -f(x) - log(p(X))
% These components (value, gradient and Hessian are used to create the
% taylor expansion)
%

%% 
% Observation model
% Y_i ~ Po(exp(z_i))
% p_Y|X(y_i|z_i) = exp(z_i*y_i)*exp(-exp(z_i))/k!

% compute log observations, and derivatives
z_0 = A*x_0;
f = y.*z_0- exp(z_0)-log( factorial(y) ); % log(p(Y|X))= log[exp(z_i*y_i)*exp(-exp(z_i))/y_i!]

% compute the function
% -log(p(X|Y)) ~ -log(p(Y|X)) - log(p(X)) = -f(x) - log(p(X)) 
neglogp = -sum(f) + 1/2 * x_0'*Q*x_0 ;

if nargout>1
  % compute derivatives (if needed, i.e. nargout>1)
  d_f = y-exp(z_0);
  d_logp = -A'*d_f +(Q+Q')*x_0;
end

if nargout>2
  % compute hessian (if needed, i.e. nargout>2)
  d2_f = -exp(z_0);
  n = size(A,1);
  d2_logp = (1/2* (Q+Q')- A'*spdiags(d2_f,0,n,n)*A);
end
