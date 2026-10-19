clear; close all; clc;

%% =========================================================
%  Physics Baseline Model — Park & Ramirez (1988) Fed-Batch
%  Pure ODE integration with known kinetic parameters.
%  Used as benchmark against PINN and hybrid models.
%
%  State variables: X (biomass), S (substrate),
%                   Pt (total protein), Pm (secreted protein), V (volume)
%
%  Kinetics (Park & Ramirez 1988):
%    mu    = 21.87*S / ((S+0.4)*(S+62.5))
%    theta = 4.75*mu / (0.12+mu)
%    fp    = S*exp(-5*S) / (0.1+S)
%
%  ODEs:
%    dX/dt  = mu*X - D*X
%    dS/dt  = -Y*mu*X + D*(Sin-S)
%    dPt/dt = fp*X - D*Pt
%    dPm/dt = theta*(Pt-Pm) - D*Pm
%    dV/dt  = F
% =========================================================

%% 1. Load data
data = readtable(fullfile('data', 'data.csv'));
data = data(~isnan(data.time_hr_), :);

t_all    = data.time_hr_;
X_true   = data.X_True_g_L_;
S_true   = data.S_True_g_L_;
Pt_true  = data.PT_True_g_L_;
Pm_true  = data.PM_True_g_L_;
V_meas   = data.V_noise_L_;
F_all    = data.Feed_L_h_;
batch_num = data.batchNumber;

% Model parameters
Sin = 20;    % substrate feed concentration (g/L)
Y   = 7.3;   % substrate/biomass yield coefficient

%% 2. Same batch split as PINN models
test_batch    = 16;
train_batches = [1, 3, 4, 7, 8, 11, 12, 13, 14, 15];
val_batches   = [2, 5, 6, 9, 10];
all_batches   = [train_batches, val_batches, test_batch];

train_idx = ismember(batch_num, train_batches);
val_idx   = ismember(batch_num, val_batches);
test_idx  = (batch_num == test_batch);

%% 3. Simulate all batches via ODE45
R2 = @(y, p) 1 - sum((y-p).^2) / (sum((y-mean(y)).^2) + 1e-10);

% Storage for all predictions
X_pred_all  = zeros(size(t_all));
S_pred_all  = zeros(size(t_all));
Pt_pred_all = zeros(size(t_all));
Pm_pred_all = zeros(size(t_all));
V_pred_all  = zeros(size(t_all));

fprintf('Simulating %d batches with Park & Ramirez ODE model...\n\n', length(all_batches));

for b = all_batches
    bidx = find(batch_num == b);
    t_b  = t_all(bidx);
    F_b  = F_all(bidx);

    % Sort by time
    [t_b, order] = sort(t_b);
    bidx_sorted  = bidx(order);
    F_b          = F_b(order);

    % Initial conditions from data (t=0 point)
    X0  = X_true(bidx_sorted(1));
    S0  = S_true(bidx_sorted(1));
    Pt0 = Pt_true(bidx_sorted(1));
    Pm0 = Pm_true(bidx_sorted(1));
    V0  = V_meas(bidx_sorted(1));
    ic  = [X0; S0; Pt0; Pm0; V0];

    % Build piecewise-constant feed interpolant
    F_fn = @(t_query) interp1(t_b, F_b, t_query, 'previous', 'extrap');

    % ODE function
    ode_fn = @(t, z) parkRamirez(t, z, F_fn, Sin, Y);

    % Integrate over the batch time span
    opts   = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 0.1);
    [~, Z] = ode45(ode_fn, t_b, ic, opts);

    % Store predictions (restore original row order)
    X_pred_all(bidx_sorted)  = Z(:,1);
    S_pred_all(bidx_sorted)  = Z(:,2);
    Pt_pred_all(bidx_sorted) = Z(:,3);
    Pm_pred_all(bidx_sorted) = Z(:,4);
    V_pred_all(bidx_sorted)  = Z(:,5);
end

%% 4. Evaluate R² per split
fprintf('=== Training set (%d batches) ===\n', length(train_batches));
fprintf('  X:  R²=%.4f\n', R2(X_true(train_idx),  X_pred_all(train_idx)));
fprintf('  S:  R²=%.4f\n', R2(S_true(train_idx),  S_pred_all(train_idx)));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(train_idx), Pt_pred_all(train_idx)));
fprintf('  Pm: R²=%.4f\n', R2(Pm_true(train_idx), Pm_pred_all(train_idx)));
fprintf('  V:  R²=%.4f\n', R2(V_meas(train_idx),  V_pred_all(train_idx)));

fprintf('\n=== Validation set (%d batches) ===\n', length(val_batches));
fprintf('  X:  R²=%.4f\n', R2(X_true(val_idx),  X_pred_all(val_idx)));
fprintf('  S:  R²=%.4f\n', R2(S_true(val_idx),  S_pred_all(val_idx)));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(val_idx), Pt_pred_all(val_idx)));
fprintf('  Pm: R²=%.4f\n', R2(Pm_true(val_idx), Pm_pred_all(val_idx)));
fprintf('  V:  R²=%.4f\n', R2(V_meas(val_idx),  V_pred_all(val_idx)));

fprintf('\n=== Test set (Batch %d — optimal control) ===\n', test_batch);
X_r2_test  = R2(X_true(test_idx),  X_pred_all(test_idx));
S_r2_test  = R2(S_true(test_idx),  S_pred_all(test_idx));
Pt_r2_test = R2(Pt_true(test_idx), Pt_pred_all(test_idx));
Pm_r2_test = R2(Pm_true(test_idx), Pm_pred_all(test_idx));
V_r2_test  = R2(V_meas(test_idx),  V_pred_all(test_idx));
fprintf('  X:  R²=%.4f\n', X_r2_test);
fprintf('  S:  R²=%.4f\n', S_r2_test);
fprintf('  Pt: R²=%.4f\n', Pt_r2_test);
fprintf('  Pm: R²=%.4f\n', Pm_r2_test);
fprintf('  V:  R²=%.4f\n', V_r2_test);

%% 5. Summary comparison table (for report)
fprintf('\n=== Summary: Physics Baseline vs PINN (Test Batch %d) ===\n', test_batch);
fprintf('%-12s  %-18s  %-18s\n', 'Variable', 'Physics Baseline', 'PINN v13 (ref)');
fprintf('%s\n', repmat('-', 1, 52));
% PINN reference values from v13 run — update these after running PINN
pinn_ref = struct('X', NaN, 'S', NaN, 'Pt', NaN, 'Pm', NaN, 'V', 1.0);
fprintf('%-12s  R²=%-15.4f  R²=%-15s\n', 'X',  X_r2_test,  '(run PINN)');
fprintf('%-12s  R²=%-15.4f  R²=%-15s\n', 'S',  S_r2_test,  '(run PINN)');
fprintf('%-12s  R²=%-15.4f  R²=%-15s\n', 'Pt', Pt_r2_test, '(run PINN)');
fprintf('%-12s  R²=%-15.4f  R²=%-15s\n', 'Pm', Pm_r2_test, '(run PINN)');
fprintf('%-12s  R²=%-15.4f  R²=%-15s\n', 'V',  V_r2_test,  '1.0000 (exact)');

%% 6. Visualization — Test batch time series
t_test      = t_all(test_idx);
[t_test_s, s_ord] = sort(t_test);

figure('Position', [100 100 1100 800], 'Color', 'w');
lw = 2.2; fs = 11;

vars = {'X','S','Pt','Pm','V'};
true_vals = {X_true(test_idx),  S_true(test_idx),  ...
             Pt_true(test_idx), Pm_true(test_idx),  V_meas(test_idx)};
pred_vals = {X_pred_all(test_idx),  S_pred_all(test_idx),  ...
             Pt_pred_all(test_idx), Pm_pred_all(test_idx), V_pred_all(test_idx)};
ylabels   = {'Biomass (g/L)','Substrate (g/L)', ...
             'Total Protein (g/L)','Secreted Protein (g/L)','Volume (L)'};
panels    = {'(A)','(B)','(C)','(D)','(E)'};

r2_test_all = [X_r2_test, S_r2_test, Pt_r2_test, Pm_r2_test, V_r2_test];

for i = 1:5
    subplot(3,2,i); hold on;
    tv = true_vals{i}; pv = pred_vals{i};
    [~, so] = sort(t_test);
    plot(t_test(so), tv(so), 'k-',  'LineWidth', lw);
    plot(t_test(so), pv(so), 'b--', 'LineWidth', lw);
    grid on; box on;
    xlabel('Time (h)', 'FontSize', fs);
    ylabel(ylabels{i}, 'FontSize', fs);
    title(sprintf('%s  R²=%.4f', panels{i}, r2_test_all(i)), 'FontSize', fs);
end

% Panel F: feed rate profile
subplot(3,2,6); hold on;
[~, so] = sort(t_test);
F_test = F_all(test_idx);
stairs(t_test(so), F_test(so), 'm', 'LineWidth', lw);
grid on; box on;
xlabel('Time (h)', 'FontSize', fs);
ylabel('Feed rate (L/h)', 'FontSize', fs);
title('(F) Feed profile', 'FontSize', fs);

subplot(3,2,1);
legend({'True (noisy)', 'Physics ODE'}, 'Location', 'best', 'FontSize', fs);

sgtitle(sprintf('Physics Baseline — Test Batch %d (Park & Ramirez ODE)', test_batch), ...
    'FontSize', 13, 'FontWeight', 'bold');

%% 7. Parity plots — all batches
figure('Position', [100 100 1200 800], 'Color', 'w');

all_idx = ismember(batch_num, all_batches);
colors  = zeros(sum(all_idx), 3);
colors(train_idx(all_idx), :) = repmat([0.2 0.4 0.8], sum(train_idx), 1);
colors(val_idx(all_idx),   :) = repmat([0.9 0.3 0.3], sum(val_idx),   1);
colors(test_idx(all_idx),  :) = repmat([0.1 0.7 0.2], sum(test_idx),  1);

parity_vars = {
    X_true(all_idx),  X_pred_all(all_idx),  'X (g/L)',  [0 3];
    S_true(all_idx),  S_pred_all(all_idx),  'S (g/L)',  [0 20];
    Pt_true(all_idx), Pt_pred_all(all_idx), 'Pt (g/L)', [0 5];
    Pm_true(all_idx), Pm_pred_all(all_idx), 'Pm (g/L)', [0 4]};

split_labels = {'Train','Val','Test'};
split_colors = {[0.2 0.4 0.8],[0.9 0.3 0.3],[0.1 0.7 0.2]};
split_idx    = {train_idx, val_idx, test_idx};

for i = 1:4
    subplot(2,2,i); hold on;
    tv = parity_vars{i,1}; pv = parity_vars{i,2};
    lims = parity_vars{i,4};
    for s = 1:3
        si = split_idx{s}(all_idx);
        scatter(tv(si), pv(si), 35, split_colors{s}, 'filled', ...
            'MarkerFaceAlpha', 0.6, 'DisplayName', ...
            sprintf('%s (R²=%.3f)', split_labels{s}, R2(tv(si), pv(si))));
    end
    plot(lims, lims, 'k--', 'LineWidth', 2, 'HandleVisibility', 'off');
    xlim(lims); ylim(lims);
    xlabel(sprintf('True %s', parity_vars{i,3}));
    ylabel(sprintf('Predicted %s', parity_vars{i,3}));
    title(parity_vars{i,3}); legend('Location','best'); grid on;
end
sgtitle('Physics Baseline — Parity Plots (All Batches)', ...
    'FontSize', 13, 'FontWeight', 'bold');

%% =========================================================
%  ODE function: Park & Ramirez fed-batch bioreactor
% =========================================================
function dzdt = parkRamirez(t, z, F_fn, Sin, Y)
    X  = max(z(1), 0);
    S  = max(z(2), 0);
    Pt = max(z(3), 0);
    Pm = max(z(4), 0);
    V  = max(z(5), 0.01);

    F = F_fn(t);
    D = F / V;

    S_safe = max(S, 1e-8);
    mu    = 21.87 * S_safe / ((S_safe + 0.4) * (S_safe + 62.5));
    theta = 4.75  * mu    / (0.12 + mu);
    fp    = S_safe * exp(-5 * S_safe) / (0.1 + S_safe);

    dX  = mu*X  - D*X;
    dS  = -Y*mu*X + D*(Sin - S);
    dPt = fp*X  - D*Pt;
    dPm = theta*(Pt - Pm) - D*Pm;
    dV  = F;

    dzdt = [dX; dS; dPt; dPm; dV];
end