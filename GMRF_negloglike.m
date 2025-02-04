function negloglike = GMRF_negloglike(theta, y, A, B, spde, qbeta, alpha)
% GMRF_NEGLOGLIKE Calculate the GMRF data likelihood, non-Gaussian observations
%
% negloglike = GMRF_negloglike(theta, y, A_tilde, spde, qbeta, alpha)
%
% theta = transformed version of tau and kappa2
% y = the data vector, as a column with n elements
% A = the observation matrix, sparse n-by-N
% B = covariate matrix, sparse n-by-Nbeta
% spde = structure with C,G,G2 matrices used to build precision matrix
% qbeta = Precision matrix for the regression parameters (scalar)
% alpha = Order of the field (1,2,...)
%

%% 

% extract parameters
tau = exp(theta(1));
kappa2 = exp(theta(2));

if alpha==1
  Q = tau*(kappa2*spde.C+ spde.G);
else
  Q = tau*(kappa2^2*spde.C+2*kappa2*spde.G+spde.G2);
end

% combine Q and Qbeta and create observation matrix
Qall = blkdiag(Q, qbeta*speye(size(B,2)));
Aall = [A B];

% declare x_mode as global so that we start subsequent optimisations from
% the previous mode (speeds up nested optimisation).
global x_mode;
if isempty(x_mode)
  % no existing mode, use zero init vector
  x_mode = zeros(size(Qall,1),1); %Qall + Aall'*Aall)\(Aall'*log(y+.1));
end

% find mode
x_mode = fminNR(@(x) GMRF_taylor(x,y,Aall,Qall), x_mode);

% find the Laplace approximation of the denominator, 
% -log(p(X|Y)) ~ -log(p(Y|X)) - log(p(X)) = -f(x) - log(p(X))
[neg_logp_x_y, ~, Q_xy] = GMRF_taylor(x_mode, y, Aall, Qall);
% note that f = -log_obs + x_mode'*Q*x_mode/2.

% Compute cholesky factors, the second output p is a numerical check of if
% the cholesky computation converged.
[R_x,p_x] = chol(Qall); 
% TODO: the code from the teatcher had Q and not Qall maybe this does not matter since we only need the determinant and the difference is small? (dim Q approx dim Qall)
[R_xy,p_xy] = chol(Q_xy);
if p_x~=0 || p_xy~=0
  % cholesky factor fail -> (almost) semidefinite matrix ->
  % -> det(Q) ~ 0 -> log(det(Q)) ~ -inf -> negloglike ~ inf
  % Set negloglike to a REALLY big value
  negloglike = realmax;
  fprintf('cholesky factor fail')
  return;
end
% compute negative log-likelihood
negloglike = (neg_logp_x_y -sum(log(diag(R_x)))+ sum(log(diag(R_xy))));
% Assumed uniform prior on theta
% print diagnostic information (progress)
fprintf(1, 'Theta: %s; negloglike: %12.4e\n', sprintf('%12.4e', theta'), negloglike);

