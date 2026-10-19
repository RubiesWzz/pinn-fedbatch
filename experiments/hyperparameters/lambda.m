clear; close all; clc;

%% ========================= 1. Data Loading =========================
data = readtable(fullfile('data', 'data.csv'));
data = data(~isnan(data.time_hr_), :);

t_all     = data.time_hr_;
X_meas    = data.X_noise_g_L_;
S_meas    = data.S_noise_g_L_;
Pt_meas   = data.PT_noise_g_L_;
Pm_meas   = data.PM_noise_g_L_;
V_meas    = data.V_noise_L_;
F_all     = data.Feed_L_h_;
batch_num = data.batchNumber;
X_true    = data.X_True_g_L_;
S_true    = data.S_True_g_L_;
Pt_true   = data.PT_True_g_L_;
Pm_true   = data.PM_True_g_L_;

Sin = 20;
Y   = 7.3;

%% ========================= 2. Data Split =========================
test_batch        = 16;
train_batches_all = [1, 3, 4, 7, 8, 11, 12, 13, 14, 15];
train_batches     = setdiff(train_batches_all, test_batch);
val_batches       = [2, 5, 6, 9, 10];

train_idx = ismember(batch_num, train_batches);
val_idx   = ismember(batch_num, val_batches);
test_idx  = (batch_num == test_batch);

t_train  = t_all(train_idx);
X_train  = X_meas(train_idx);
S_train  = S_meas(train_idx);
Pt_train = Pt_meas(train_idx);
Pm_train = Pm_meas(train_idx);
V_train  = V_meas(train_idx);
F_train  = F_all(train_idx);

%% ========================= 3. Normalisation =========================
stats.t  = [min(t_train),  max(t_train)];
stats.X  = [min(X_train),  max(X_train)];
stats.S  = [0, max(S_train)];
stats.V  = [0, 25];
stats.F  = [0, 2.5];
stats.Pt = [0, max(Pt_train)];
stats.Pm = [0, max(Pm_train)];

norm_fn = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);

%% ========================= 4. V_exact =========================
all_batches_sorted = unique([train_batches, val_batches, test_batch]);
V_exact_all = zeros(size(t_all));
for b = all_batches_sorted
    bidx = find(batch_num == b);
    t_b = t_all(bidx); F_b = F_all(bidx); V_b = V_meas(bidx);
    [~, order] = sort(t_b);
    bidx = bidx(order); t_b = t_b(order); F_b = F_b(order); V_b = V_b(order);
    V_int = zeros(length(t_b),1);
    V_int(1) = V_b(1);
    for k = 2:length(t_b)
        V_int(k) = V_int(k-1) + F_b(k-1)*(t_b(k)-t_b(k-1));
    end
    V_exact_all(bidx) = V_int;
end

V_exact_train = V_exact_all(train_idx);
V_exact_val   = V_exact_all(val_idx);

%% ========================= 5. Prepare dlarray inputs =========================
build_input = @(t, X, S, Pt, Pm, V, F) [
    norm_fn(t,  stats.t), ...
    norm_fn(X,  stats.X), ...
    norm_fn(S,  stats.S), ...
    norm_fn(Pt, stats.Pt), ...
    norm_fn(Pm, stats.Pm), ...
    norm_fn(V,  stats.V), ...
    norm_fn(F,  stats.F)
]';

inputTrain = build_input(t_train, X_train, S_train, Pt_train, Pm_train, ...
                         V_exact_train, F_train);
dlInputTrain = dlarray(single(inputTrain), 'CB');

inputVal = build_input(t_all(val_idx), X_meas(val_idx), S_meas(val_idx), ...
    Pt_meas(val_idx), Pm_meas(val_idx), V_exact_val, F_all(val_idx));
dlInputVal = dlarray(single(inputVal), 'CB');

targets.X  = norm_fn(X_train,  stats.X);
targets.S  = norm_fn(S_train,  stats.S);
targets.Pt = norm_fn(Pt_train, stats.Pt);
targets.Pm = norm_fn(Pm_train, stats.Pm);

targets_val.X  = norm_fn(X_meas(val_idx),  stats.X);
targets_val.S  = norm_fn(S_meas(val_idx),  stats.S);
targets_val.Pt = norm_fn(Pt_meas(val_idx), stats.Pt);
targets_val.Pm = norm_fn(Pm_meas(val_idx), stats.Pm);
%% ========================= 6. Lambda Search Settings =========================
lambda_values = [0.0, 0.25, 0.5, 0.75, 0.9];
nLambda       = length(lambda_values);

% Training hyperparameters (same as v12M but FIXED lambda — no staged schedule)
totalEpochs  = 5000;   % Reduced for the search; increase for final runs
learningRate = 0.001;
numColloc    = 350;
n_batches_per_epoch = round(length(train_batches) * 0.67);

% Fixed weighting used across all lambda runs (mid-stage values from v12M)
Pm_phys_w = 1.5;
S_phys_w  = 1.5;

%% ========================= 7. Network Builder =========================
numInputs  = 7;
numOutputs = 4;
hiddenSize = 50;

build_net = @() dlnetwork([
    featureInputLayer(numInputs, 'Normalization', 'none')
    fullyConnectedLayer(hiddenSize); tanhLayer
    fullyConnectedLayer(hiddenSize); tanhLayer
    fullyConnectedLayer(hiddenSize); tanhLayer
    fullyConnectedLayer(hiddenSize); tanhLayer
    fullyConnectedLayer(numOutputs)
    sigmoidLayer]);

%% ========================= 8. Pre-allocate results =========================
res.train_data   = zeros(1, nLambda);
res.train_phys   = zeros(1, nLambda);
res.val_data     = zeros(1, nLambda);
res.val_phys     = zeros(1, nLambda);

% Prepare per-batch indices — indices into the FULL train set (columns of dlInputTrain)
% batch_num(train_idx) gives the batch labels for each training point in order,
% so find() returns positions within that sub-vector, which directly index dlInputTrain columns.
batch_labels = batch_num(train_idx);   % batch label for every training point
batch_train_idx = cell(length(train_batches), 1);
for i = 1:length(train_batches)
    batch_train_idx{i} = find(batch_labels == train_batches(i));
end
% Sanity check
total_pts = sum(cellfun(@numel, batch_train_idx));
assert(total_pts == sum(train_idx), ...
    'Batch index mismatch: %d indexed vs %d training points', total_pts, sum(train_idx));

fprintf('==========================================\n');
fprintf(' Lambda search: %d values × %d epochs\n', nLambda, totalEpochs);
fprintf('==========================================\n\n');

%% ========================= 9. Main Loop =========================
for li = 1:nLambda
    lam = lambda_values(li);
    fprintf('[%d/%d] λ = %.2f  — training...\n', li, nLambda, lam);

    % Fresh network + optimiser state for every lambda
    net      = build_net();
    avgGrad  = [];
    avgSqGrad = [];

    tic;
    for epoch = 1:totalEpochs
        % Stochastic batch selection
        sel_ids = randperm(length(train_batches), n_batches_per_epoch);
        ep_idx  = [];
        for i = sel_ids
            ep_idx = [ep_idx; batch_train_idx{i}]; %#ok<AGROW>
        end

        dlIn_ep = dlInputTrain(:, ep_idx);
        targ_ep.X  = targets.X(ep_idx);
        targ_ep.S  = targets.S(ep_idx);
        targ_ep.Pt = targets.Pt(ep_idx);
        targ_ep.Pm = targets.Pm(ep_idx);

        colloc = genColloc(numColloc, t_train, X_train, S_train, ...
            Pt_train, Pm_train, V_train, F_train, stats);

        [loss, grads, ~, ~] = dlfeval(@lossFn, net, dlIn_ep, ...
            targ_ep, colloc, lam, Pm_phys_w, S_phys_w, stats, Sin, Y);

        [net, avgGrad, avgSqGrad] = adamupdate(net, grads, avgGrad, ...
            avgSqGrad, epoch, learningRate);
    end
    elapsed = toc;

    % ---- Evaluate final losses on full train and val sets ----
    colloc_eval = genColloc(numColloc*3, t_train, X_train, S_train, ...
        Pt_train, Pm_train, V_train, F_train, stats);

    % Train data loss (full set, no stochastic sub-sampling)
    [~, ~, dL_tr, pL_tr] = dlfeval(@lossFn, net, dlInputTrain, targets, ...
        colloc_eval, lam, Pm_phys_w, S_phys_w, stats, Sin, Y);

    % Validation data loss (reuse same colloc for physics)
    [~, ~, dL_val, pL_val] = dlfeval(@lossFn, net, dlInputVal, targets_val, ...
        colloc_eval, lam, Pm_phys_w, S_phys_w, stats, Sin, Y);

    res.train_data(li) = double(dL_tr);
    res.train_phys(li) = double(pL_tr);
    res.val_data(li)   = double(dL_val);
    res.val_phys(li)   = double(pL_val);

    fprintf('         Done (%.0fs) | Train: D=%.4f P=%.4f | Val: D=%.4f P=%.4f\n\n', ...
        elapsed, res.train_data(li), res.train_phys(li), res.val_data(li), res.val_phys(li));
end

%% ========================= 10. Save Results =========================
save('lambda_search_results.mat', 'lambda_values', 'res');
fprintf('Results saved to lambda_search_results.mat\n\n');

%% ========================= 11. Bar Chart (Fig. 6B style) =========================
plot_lambda_bar(lambda_values, res);

%% ========================= Helper: Bar Chart Function =========================
function plot_lambda_bar(lambda_values, res)
    nL = length(lambda_values);

    % Colours matching the paper (Fig. 6)
    c_train_data  = [0.20, 0.40, 0.75];   % dark blue
    c_train_phys  = [0.55, 0.85, 0.95];   % light blue / cyan
    c_val_data    = [0.80, 0.35, 0.30];   % salmon/red
    c_val_phys    = [0.75, 0.35, 0.75];   % purple/magenta

    % Group positions: 4 bars per lambda, small gap between groups
    bar_w     = 0.18;
    offsets   = [-1.5, -0.5, 0.5, 1.5] * bar_w;
    x_centers = 1:nL;

    fig = figure('Position', [100 100 900 520], 'Color', 'w');
    ax  = axes('Parent', fig);
    hold(ax, 'on');

    for li = 1:nL
        xc = x_centers(li);
        vals = [res.train_data(li), res.train_phys(li), ...
                res.val_data(li),   res.val_phys(li)];
        colors = {c_train_data, c_train_phys, c_val_data, c_val_phys};
        for bi = 1:4
            bar(ax, xc + offsets(bi), vals(bi), bar_w, ...
                'FaceColor', colors{bi}, 'EdgeColor', 'none');
        end
    end

    % Log scale y-axis
    set(ax, 'YScale', 'log');
    ylim(ax, [1e-5, 1]);

    % x-axis ticks and labels
    set(ax, 'XTick', x_centers, ...
            'XTickLabel', arrayfun(@(v) sprintf('%.2g', v), lambda_values, ...
                                   'UniformOutput', false));
    xlabel(ax, 'Lambda (\lambda)', 'FontSize', 13);
    ylabel(ax, 'Loss value (normalized)', 'FontSize', 13);
    title(ax, 'Effect of \lambda on Train / Validation Losses', 'FontSize', 14, 'FontWeight', 'bold');

    % Legend
    h = [bar(ax, NaN, NaN, bar_w, 'FaceColor', c_train_data, 'EdgeColor', 'none'), ...
         bar(ax, NaN, NaN, bar_w, 'FaceColor', c_train_phys, 'EdgeColor', 'none'), ...
         bar(ax, NaN, NaN, bar_w, 'FaceColor', c_val_data,   'EdgeColor', 'none'), ...
         bar(ax, NaN, NaN, bar_w, 'FaceColor', c_val_phys,   'EdgeColor', 'none')];
    legend(h, {'Train data loss', 'Train physics loss', ...
               'Validation data loss', 'Validation physics loss'}, ...
           'Location', 'northeast', 'FontSize', 10);

    grid(ax, 'on');
    box(ax, 'on');
    set(ax, 'GridAlpha', 0.3);

    % Mark best lambda with annotation
    [~, best_li] = min(res.val_data);
    xb = x_centers(best_li);
    yb = max([res.val_data(best_li), res.val_phys(best_li)]) * 3;
    text(ax, xb, yb, sprintf('Best\nλ=%.2g', lambda_values(best_li)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.1 0.5 0.1], ...
        'FontWeight', 'bold');

    hold(ax, 'off');
    set(fig, 'Color', 'w');

    % Save figure
    saveas(fig, 'lambda_search_barplot.png');
    fprintf('Figure saved to lambda_search_barplot.png\n');

    % Print summary table
    fprintf('\n========= Lambda Search Summary =========\n');
    fprintf('  λ      Train-D   Train-P   Val-D     Val-P\n');
    for li = 1:length(lambda_values)
        fprintf('  %.2f   %.4f    %.4f    %.4f    %.4f\n', ...
            lambda_values(li), res.train_data(li), res.train_phys(li), ...
            res.val_data(li), res.val_phys(li));
    end
    [~, bi] = min(res.val_data);
    fprintf('\n  --> Best lambda (lowest val data loss): λ = %.2f\n', lambda_values(bi));
    fprintf('=========================================\n');
end

%% ========================= Loss Function =========================
function [loss, grads, dL, pL] = lossFn(net, dlX, targets, colloc, lambda, Pm_phys_w, S_phys_w, stats, Sin, Y)
    dlY = forward(net, dlX);
    w   = [1, 1, 2, 5];
    dL  = w(1)*mean((dlY(1,:) - targets.X' ).^2) + ...
          w(2)*mean((dlY(2,:) - targets.S' ).^2) + ...
          w(3)*mean((dlY(3,:) - targets.Pt').^2) + ...
          w(4)*mean((dlY(4,:) - targets.Pm').^2);
    dL  = dL / sum(w);

    pL   = physLoss(net, colloc, Pm_phys_w, S_phys_w, stats, Sin, Y);
    loss = (1 - lambda)*dL + lambda*pL;
    grads = dlgradient(loss, net.Learnables);
    dL    = extractdata(dL);
    pL    = extractdata(pL);
end

%% ========================= Physics Loss =========================
function pL = physLoss(net, c, Pm_phys_w, S_phys_w, stats, Sin, Y)
    dlC = dlarray(single(c.input), 'CB');
    dlY = forward(net, dlC);

    denorm = @(y, s) y .* (s(2) - s(1)) + s(1);
    X  = denorm(dlY(1,:), stats.X);
    S  = denorm(dlY(2,:), stats.S);
    Pt = denorm(dlY(3,:), stats.Pt);
    Pm = denorm(dlY(4,:), stats.Pm);
    V  = dlarray(single(c.V_raw), 'CB');
    Fr = c.F_raw;
    D  = Fr ./ max(V, 0.1);

    S_safe = max(S, 1e-6);
    mu     = 21.87 * S_safe ./ ((S_safe + 0.4) .* (S_safe + 62.5));
    theta  = 4.75  * mu     ./ (0.12 + mu);
    fp     = S_safe .* exp(-5 * S_safe) ./ (0.1 + S_safe);

    dt      = 0.01;
    dt_norm = dt / (stats.t(2) - stats.t(1));
    cPlus   = c.input;
    cPlus(1,:) = cPlus(1,:) + dt_norm;
    dlCp   = dlarray(single(cPlus), 'CB');
    dlYp   = forward(net, dlCp);

    dX  = (dlYp(1,:) - dlY(1,:)) * (stats.X(2)  - stats.X(1))  / dt;
    dS  = (dlYp(2,:) - dlY(2,:)) * (stats.S(2)  - stats.S(1))  / dt;
    dPt = (dlYp(3,:) - dlY(3,:)) * (stats.Pt(2) - stats.Pt(1)) / dt;
    dPm = (dlYp(4,:) - dlY(4,:)) * (stats.Pm(2) - stats.Pm(1)) / dt;

    rX  = dX  - (mu.*X  - D.*X);
    rS  = dS  - (-Y*mu.*X + D.*(Sin - S));
    rPt = dPt - (fp.*X  - D.*Pt);
    rPm = dPm - (theta.*(Pt - Pm) - D.*Pm);

    w_phys = [1, S_phys_w, 1.5, Pm_phys_w];
    pL = (w_phys(1)*(rX  / (stats.X(2)  - stats.X(1)) ).^2 + ...
          w_phys(2)*(rS  / (stats.S(2)  - stats.S(1)) ).^2 + ...
          w_phys(3)*(rPt / (stats.Pt(2) - stats.Pt(1))).^2 + ...
          w_phys(4)*(rPm / (stats.Pm(2) - stats.Pm(1))).^2);
    pL = mean(pL, 'all') / sum(w_phys);
end

%% ========================= Collocation Generator =========================
function c = genColloc(n, t, X, S, Pt, Pm, V, F, stats)
    idx   = randi(length(t), 1, n);
    t_raw = t(idx)' + (rand(1,n) - 0.5)*0.5;
    t_raw = max(min(t_raw, max(t)), min(t));
    V_raw = max(V(idx)', 0.1);
    F_raw = F(idx)';

    X_raw  = stats.X(1)  + (stats.X(2)  - stats.X(1))  * rand(1,n);
    S_raw  = stats.S(1)  + (stats.S(2)  - stats.S(1))  * rand(1,n);
    Pt_raw = stats.Pt(1) + (stats.Pt(2) - stats.Pt(1)) * rand(1,n);
    Pm_raw = stats.Pm(1) + (stats.Pm(2) - stats.Pm(1)) * rand(1,n);

    c.t_raw = t_raw; c.V_raw = V_raw; c.F_raw = F_raw;
    norm_fn = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);
    c.input = [
        norm_fn(t_raw,  stats.t);
        norm_fn(X_raw,  stats.X);
        norm_fn(S_raw,  stats.S);
        norm_fn(Pt_raw, stats.Pt);
        norm_fn(Pm_raw, stats.Pm);
        norm_fn(V_raw,  stats.V);
        norm_fn(F_raw,  stats.F)
    ];
end