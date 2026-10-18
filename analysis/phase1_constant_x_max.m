%% main_logistic_fedbatch.m
clear; clc; close all;
rng(1); % 固定随机种子便于复现

%% -------------------------
% 1) Ground-truth parameters (用于造数据)
% --------------------------
theta_true.mu_max = 0.3;    % h^-1
theta_true.X_max  = 47.3;   % g/L

%% -------------------------
% 2) Experiment design (你给的范围)
% --------------------------
X0 = 1.5;     % g/L  (可在 [0.5, 2.5] 内改)
V0 = 2.5;     % L    (可在 [1.9, 3.5] 内改)

T_end = 24;   % h  (你也可以改成 30/40，看是否更接近Xmax)
dt_meas = 1;  % 1小时采样
t_meas = (0:dt_meas:T_end)';

% feed profile: piecewise constant, values in [0.05, 0.5] L/h
% 你可以改成你想要的分段结构
F_profile = struct();
F_profile.t_breaks = [0, 8, 16, 24];  % 分段区间端点（必须从0到T_end）
F_profile.F_values = [0.10, 0.30, 0.15]; % 每段的F (长度 = breaks-1)

%% -------------------------
% 3) Simulate "true" trajectory and generate noisy measurements
% --------------------------
noise_rel = 0.05; % 5% relative Gaussian noise

[data] = simulate_and_sample(theta_true, X0, V0, F_profile, t_meas, noise_rel);

% data fields:
% data.t, data.X_true, data.V_true, data.X_meas

%% -------------------------
% 4) Parameter estimation (classic nonlinear least squares)
% --------------------------
% Unknowns: [mu_max, X_max]
% x0_guess = [0.2, 47.3];       % 初值（可改，多初值更稳）
% lb = [1e-4, 1e-2];          % 参数下界：正数
% ub = [2.0, 200];            % 参数上界：给个合理范围

Xmax_fixed = 47.3;        % 固定真值（你要验证的假设）
mu0_guess = 0.2;          % 只需要 mu 的初值
lb = 1e-4;
ub = 2.0;

opts = optimoptions('lsqnonlin', ...
    'Display','iter', ...
    'MaxFunctionEvaluations', 5000, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-10);

% residual_fun = @(p) residual_X(p, X0, V0, F_profile, t_meas, data.X_meas);
residual_fun = @(mu) residual_mu_only(mu, X0, V0, F_profile, t_meas, data.X_meas, Xmax_fixed);
[mu_hat, resnorm, residual, exitflag, output] = lsqnonlin(residual_fun, mu0_guess, lb, ub, opts);
fprintf('\nEstimated parameter (Xmax fixed):\n');
fprintf('mu_max_hat = %.6f (true %.6f)\n', mu_hat, theta_true.mu_max);

theta_hat.mu_max = mu_hat;
theta_hat.X_max  = Xmax_fixed;

sim_hat = simulate_and_sample(theta_hat, X0, V0, F_profile, t_meas, 0.0);

% [p_hat, resnorm, residual, exitflag, output] = lsqnonlin(residual_fun, x0_guess, lb, ub, opts);
% 
% mu_hat = p_hat(1);
% Xmax_hat = p_hat(2);
% 
% fprintf('\nEstimated parameters:\n');
% fprintf('mu_max_hat = %.6f (true %.6f)\n', mu_hat, theta_true.mu_max);
% fprintf('X_max_hat  = %.6f (true %.6f)\n', Xmax_hat, theta_true.X_max);

%% -------------------------
% 5) Plot fit
% --------------------------
theta_hat.mu_max = mu_hat;
theta_hat.X_max  = 47.3;

sim_hat = simulate_and_sample(theta_hat, X0, V0, F_profile, t_meas, 0.0); % 无噪声拟合轨迹

figure;
plot(t_meas, data.X_meas, 'o', 'DisplayName','Measured X (noisy)'); hold on;
plot(t_meas, data.X_true, '-', 'LineWidth', 1.5, 'DisplayName','True X');
plot(t_meas, sim_hat.X_true, '--', 'LineWidth', 1.5, 'DisplayName','Fitted X');
grid on; xlabel('t (h)'); ylabel('X (g/L)');
legend('Location','best');
title('Logistic-fedbatch: data vs fit');

figure;
plot(t_meas, data.V_true, '-', 'LineWidth', 1.5); grid on;
xlabel('t (h)'); ylabel('V (L)');
title('Volume trajectory');

%% -------------------------
% 6) (可选) sanity checks: show D(t)=F/V
% --------------------------
D = arrayfun(@(ti) feed_rate(ti, F_profile) ./ interp1(t_meas, data.V_true, ti, 'linear', 'extrap'), t_meas);
figure;
plot(t_meas, D, '-', 'LineWidth', 1.5); grid on;
xlabel('t (h)'); ylabel('D(t) = F/V (1/h)');
title('Dilution rate trajectory');
