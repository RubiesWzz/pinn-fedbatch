%% phase2_sensitivity_fd.m
clear; clc; close all;
rng(1);

% ---- truth (或你想评估的nominal点：可以用theta_true，也可以用Phase1估计的theta_hat)
theta0.mu_max = 0.3;   % h^-1
theta0.X_max  = 47.3;  % g/L

% ---- experiment settings (与你Phase1保持一致)
X0 = 1.5;
V0 = 2.5;
T_end = 48;
t_meas = (0:1:T_end)';

F_profile.t_breaks = [0, 8, 16, 24];
F_profile.F_values = [0.10, 0.30, 0.15];

% ---- simulate nominal (no noise!)
sim0 = simulate_and_sample(theta0, X0, V0, F_profile, t_meas, 0.0);
X_nom = sim0.X_true;

% ---- finite-difference steps (相对步长更稳)
dmu   = 1e-3 * max(theta0.mu_max, 1e-6);   % 0.1% of mu_max
dXmax = 1e-3 * max(theta0.X_max,  1e-6);   % 0.1% of X_max

% ---- sensitivity wrt mu_max
theta_p = theta0; theta_m = theta0;
theta_p.mu_max = theta0.mu_max + dmu;
theta_m.mu_max = theta0.mu_max - dmu;
Xp = simulate_and_sample(theta_p, X0, V0, F_profile, t_meas, 0.0).X_true;
Xm = simulate_and_sample(theta_m, X0, V0, F_profile, t_meas, 0.0).X_true;
S_mu = (Xp - Xm) / (2*dmu);

% ---- sensitivity wrt X_max
theta_p = theta0; theta_m = theta0;
theta_p.X_max = theta0.X_max + dXmax;
theta_m.X_max = theta0.X_max - dXmax;
Xp = simulate_and_sample(theta_p, X0, V0, F_profile, t_meas, 0.0).X_true;
Xm = simulate_and_sample(theta_m, X0, V0, F_profile, t_meas, 0.0).X_true;
S_Xmax = (Xp - Xm) / (2*dXmax);

% ---- normalized sensitivities (更适合比较"谁更重要")
% 归一化定义：\tilde S_i = (theta_i / X) * dX/dtheta_i
epsX = 1e-9;
Sn_mu   = (theta0.mu_max ./ max(X_nom, epsX)) .* S_mu;
Sn_Xmax = (theta0.X_max  ./ max(X_nom, epsX)) .* S_Xmax;

% ---- influence metrics (离散时间能量)
I_mu   = sum(Sn_mu.^2);
I_Xmax = sum(Sn_Xmax.^2);

fprintf('Normalized sensitivity energy:\n');
fprintf('I_mu   = %.4e\n', I_mu);
fprintf('I_Xmax = %.4e\n', I_Xmax);
fprintf('Ratio I_mu / I_Xmax = %.4e\n', I_mu / max(I_Xmax, 1e-12));

% ---- plots
figure;
plot(t_meas, X_nom, 'LineWidth', 1.5); grid on;
xlabel('t (h)'); ylabel('X (g/L)');
title('Nominal biomass trajectory');

figure;
plot(t_meas, S_mu,   'LineWidth', 1.5); hold on;
plot(t_meas, S_Xmax, 'LineWidth', 1.5); grid on;
xlabel('t (h)'); ylabel('\partial X / \partial \theta');
legend('S_{\mu_{max}}','S_{X_{max}}','Location','best');
title('Raw sensitivities');

figure;
plot(t_meas, Sn_mu,   'LineWidth', 1.5); hold on;
plot(t_meas, Sn_Xmax, 'LineWidth', 1.5); grid on;
xlabel('t (h)'); ylabel('normalized sensitivity');
legend('\tilde S_{\mu_{max}}','\tilde S_{X_{max}}','Location','best');
title('Normalized sensitivities (comparable)');
