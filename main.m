%% Project 2: GMRF:s with Non-Gaussian Data
%
% Spatial Statistics with Image Analysis
%
% Group 13: Mohamed Abrash & Maiya Tebäck
%
% In this project, we will be studying the number of yearly coal mine
% accidents in Britain between 1750 and 1980
%
% for a clean setup, run
clear; clc;
close all;
% addpath('../functions', '../data')              % Add this line to update
% the path, if needed
%% load data
coal = readmatrix('coal.csv');

%% Mark a few decades as validation data
rng(31); % set the random seed for reproducibility
I = 175:197; % decades (so 175 -> 1750; 197 -> 1970)
I = I(randperm(length(I),5)); % pick 5 of the 23 decades as validation
I = ismember(floor(coal(:,1)/10),I); % find corresponding decades among the years.

%% Plots - marking validation data
figure(1)
scatter(coal(~I,1),coal(~I,2),30,'filled')
hold on
scatter(coal(I,1),coal(I,2),30,'filled')
hold off
title( 'Coal mining accidents data' )
legend('Modeling data', 'Validation data')
ylabel('Number of accidents per year')
xlabel('Year')

%% create precision matrices
n = size(coal,1); % number of observations
G = spdiags( repmat([-1 2 -1],[n 1]), -1:1, n, n);
C = speye(n);
% edge adjustment
G(1,1) = 1;
G(end,end) = 1;
% collect C, G and G2 matrices in an spde object
spde.C = C;
spde.G = G;
spde.G2 = G'*G;
model_flag = 1; % 1 for CAR and 0 for SAR models
%% create observations matrices
Aobs = sparse(1:sum(~I), find(~I), 1, sum(~I), n);
Avalid = sparse(1:sum(I), find(I), 1, sum(I), n);
Aall = speye(n);
% and regression basis matrices assuming just an intercept
Ball = ones(n,1);

Bobs = Aobs*Ball;
Bvalid = Avalid*Ball;
qbeta = 10^(-3); % for the beta prior

%% illustrate sparsity (since it's a temporal-process the matrices have optimal order)
figure(2)
subplot(231)
spy(spde.C)
title('C matrix')
subplot(232)
spy(spde.G)
title('G matrix')
subplot(233)
spy(spde.G2)
title('G2 matrix')
subplot(234)
spy(spde.C+2*spde.G+spde.G2)
title('C + 2*G + G2 matrix')
subplot(235)
spy(chol(spde.C+2*spde.G+spde.G2))
title('Cholesky factorisation')


%% Fit model

% we need a global variable for x_mode for the nested optimization to work
% between optimisation calls
global x_mode;
x_mode = [];
par=[0 0]; % initial parameters theta(1) = exp(tau) and theta(2)= exp(kappa^2) for the construction of the Q matrix of SAR and CAR models
%% 

% attempt to estimate parameters (optim in optim...)
% [0 0] - two parameters to estimate, lets start with 0 in log-scale.
NegloglikGMRF = @(theta) GMRF_negloglike(theta, coal(~I,2), Aobs, Bobs, spde, qbeta,model_flag);
for k = 1:1 % repeat if we need more steps for convergence (or try different initial values)
    par = fminsearch( NegloglikGMRF, par);
end
% conditional mean is now given by the mode
E_xy = x_mode;
%% Calculating posterior variance

% Q can be computed from the estimated parameters, 
tau = exp(par(1));
kappa2 = exp(par(2));
if model_flag ==1
    % CAR
    Q = blkdiag(tau*(kappa2*spde.C +spde.G), qbeta*speye(size(Bobs,2)));
    
else
    % SAR
    Q = blkdiag(tau*(kappa2^2*spde.C+2*kappa2*spde.G+spde.G2), qbeta*speye(size(Bobs,2)));
end
% use the taylor expansion to compute posterior precision
[~, ~, Q_xy] = GMRF_taylor(x_mode, coal(~I,2), [Aobs Bobs], Q);
% posterior: xtilde|y,theta = N(x_mode, Q_xy^-1)

%% Simulating Fields from the posterior
% to simulate fields from this field take x_sim = x_mode + R^-1 e, where R is
% cholesky of Q_xy and e is random vector
% then get the latent field as z = Ax and y = exp(z)
N = 10000;
R = chol(Q_xy);
ns = size(coal,1); % sample size
X = zeros(ns+1,N);
X = x_mode + R\randn(ns+1,N);
Z = [Aall Ball]*X;
% variance of y_rec by simulation
V_y_rec = var(exp(Z'));

%% Calculate the variance of exp(AX) by explicit calculation
SigmaZ = [Aall Ball]*inv(Q_xy)*[Aall Ball]';
% Variance of an exponentialnormal 
Vy = (exp(diag(SigmaZ))-ones(length(diag(SigmaZ)))).*exp(2*[Aall Ball]*x_mode +diag(SigmaZ));

%% Plots
% accident recunstructions 
y_rec = exp([Aall Ball]*E_xy);
% Intercept recontruction
y_rec_mean = exp([zeros(size(Aall)) Ball]*E_xy); 

figure(3)
hold on
% plot data
scatter(coal(~I,1),coal(~I,2),30,'filled',DisplayName='Modelling Data')
scatter(coal(I,1),coal(I,2),30,'filled',DisplayName='Validation Data')
% plot reconstruction 
scatter(coal(:,1),y_rec,30,'filled',DisplayName='Recunstruction')
% plot intercept (fixed features)
plot(coal(:,1),y_rec_mean,'LineWidth', 1,DisplayName='Intercept')

% Prediction intervals; fix theta and and get poisson quantiles
alpha = 0.2;
lambda = exp([Aall Ball]*E_xy);
lower_quantile = poissinv(alpha / 2, lambda); 
upper_quantile = poissinv(1 - alpha / 2, lambda);
plot(coal(:,1), lower_quantile,'r--',DisplayName='80% Prediction Intervals')
plot(coal(:,1), upper_quantile,'r--',DisplayName='80% Prediction Intervals')


% Standard deviations computed thru simulation. 
plot(coal(:,1), y_rec +sqrt(V_y_rec'),'b--',DisplayName='Recunstruction Standard Deviation')
plot(coal(:,1), y_rec -sqrt(V_y_rec'),'b--')

% Standard deviations thru direct computation (should overlap with the simulation)
%plot(coal(:,1), y_rec +sqrt(Vy),'k:',DisplayName='Standard Deviation')
%plot(coal(:,1), y_rec -sqrt(Vy),'k:')

hold off
title('CAR model')
legend('Modelling Data','Validation Data','Recunstruction','Intercept','80% Prediction Intervals','80% Prediction Intervals','Reconstruction Standard Deviation')
ylabel('Number of Accidents')
xlabel('Year')


%% Latent Field components

% Samples for the mean component
Z_mean = ([zeros(size(Aall)) Ball]*X);
z_mean = mean(Z_mean,2);
sd_mean = std(Z_mean')';

Z_random_field = ([Aall zeros(size(Ball))]*X);
sd_rand_field = std(Z_random_field')';
z_random_field= mean(Z_random_field,2);

C = coal(:,1);
z_upper = z_mean+sd_mean;
z_lower = z_mean-sd_mean;

z_r_upper = z_random_field + sd_rand_field;
z_r_lower = z_random_field - sd_rand_field;
%% Plot latent field components
figure;

hold on
% Plot components of the latent field
plot(C, z_mean, 'red', 'LineWidth', 1,DisplayName='Intercept');
plot(C,z_random_field,'blue','LineWidth',1,DisplayName='Random Component')

% Plot the confidence interval as a ribbon
fill([C; flipud(C)], [z_upper; flipud(z_lower)], 'red', ...
    'FaceAlpha', 0.4, 'EdgeColor', 'none');

fill([C; flipud(C)],[z_r_upper; flipud(z_r_lower)], 'blue', ...
    'FaceAlpha', 0.4, 'EdgeColor', 'none')



hold off
legend('Intercept','Random Component','Standard Deviation','Standard Deviation')
title('Simulated CAR Latent Field Components and Standard Deviation')
%% Some model evaluation
% RMSE 
% taining 
sqrt(mean((coal(~I,2) - y_rec(~I)).^2))
% validaiton
sqrt(mean((coal(I,2) - y_rec(I)).^2))


%%
figure;

scatter(coal(:,2),y_rec,'filled')
xlabel('Observations (Accidents)')
ylabel('Predicitons')

figure;

scatter(coal(:,2),V_y_rec,'filled')
xlabel('Observations (Accidents)')
ylabel('Standard Deviations of Predicitons ')
%%


%% Sensitivity analysis
dtheta = [0.01 0.1 1 10 100 1000];

% Define custom colors as RGB triplets (shades of blue)
colors = [
    0.1, 0.1, 0.1;   % Gray
    1, 0, 0;         % Red
    0, 0, 1;         % Blue
    0, 1, 0;         % Green
    1, 0.5, 0;       % Orange
    0.5, 0, 0.5      % Purple
];

figure;
C = coal(:,1);
hold on

for k = 1:6
    tau = par(1)+ dtheta(k);
    kappa2 = par(2)+dtheta(k);
    Q = blkdiag(tau*(kappa2^2*spde.C+2*kappa2*spde.G+spde.G2), qbeta*speye(size(Bobs,2)));

    N = 10000;
    R = chol(Q_xy);
    ns = size(coal,1); % sample size
    X = zeros(ns+1,N);
    X = x_mode + R\randn(ns+1,N);
    Z = [Aall Ball]*X;
    
    sdk = std(Z')';
    

% Plot components of the latent field


    plot(C, sdk, 'LineWidth', 1,DisplayName='Mean',Color=colors(k,:));

end
hold off
legend('\theta + 0.01','\theta + 0.1','\theta + 1','\theta + 10','\theta + 100')
title('Sensitivity Plot For Latent Field Standard Deviations')