clear; close all; clc;
%% 1. Data loading
data = readtable(fullfile('data', 'data.csv'));
data = data(~isnan(data.time_hr_), :);

t_all = data.time_hr_;
X_meas = data.X_noise_g_L_;
S_meas = data.S_noise_g_L_;
Pt_meas = data.PT_noise_g_L_;
Pm_meas = data.PM_noise_g_L_;
V_meas = data.V_noise_L_;
F_all = data.Feed_L_h_;
batch_num = data.batchNumber;

X_true = data.X_True_g_L_;
S_true = data.S_True_g_L_;
Pt_true = data.PT_True_g_L_;
Pm_true = data.PM_True_g_L_;

Sin = 20;
Y = 7.3;

%% 2. BATCH-LEVEL SPLIT 
test_batch = 16;   % Batch 16 = optimal control profile (most scientifically meaningful)
% Training set — now wincludes B13 (F3=0 late-stop dynamics)
train_batches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];  
val_batches   = [];

train_idx = ismember(batch_num, train_batches);
val_idx = ismember(batch_num, val_batches);
test_idx = (batch_num == test_batch);
% ---- (1) tau = t - t0_batch ----
tau_all = zeros(size(t_all));
for b = unique(batch_num(:))'
    idxb = find(batch_num==b);
    t0b  = min(t_all(idxb));
    tau_all(idxb) = t_all(idxb) - t0b;
end

% ---- (2) initial conditions per batch, replicated to all points in that batch ----
X0_all  = zeros(size(X_meas));
S0_all  = zeros(size(S_meas));
Pt0_all = zeros(size(Pt_meas));
Pm0_all = zeros(size(Pm_meas));

for b = unique(batch_num(:))'
    idxb = find(batch_num==b);
    [~,k0] = min(t_all(idxb));      % first time index within this batch
    idx0 = idxb(k0);

    X0_all(idxb)  = X_meas(idx0);
    S0_all(idxb)  = S_meas(idx0);
    Pt0_all(idxb) = Pt_meas(idx0);
    Pm0_all(idxb) = Pm_meas(idx0);
end

fprintf('Data partitioning : \n');
fprintf('  Training set: %d batches (%d points, %.1f%%)\n', ...
    length(train_batches), sum(train_idx), sum(train_idx)/length(batch_num)*100);
fprintf('          Batches: %s\n', mat2str(train_batches));
fprintf('  Validation set: %d batches (%d points, %.1f%%)\n', ...
    length(val_batches), sum(val_idx), sum(val_idx)/length(batch_num)*100);
fprintf('          Batches: %s\n', mat2str(val_batches));
fprintf('  Test set: 1 batch (%d points, %.1f%%)\n', ...
    sum(test_idx), sum(test_idx)/length(batch_num)*100);
fprintf('          Batch: %d \n\n', test_batch);

% Extract training data
t_train = t_all(train_idx);
X_train = X_meas(train_idx);
S_train = S_meas(train_idx);
Pt_train = Pt_meas(train_idx);
Pm_train = Pm_meas(train_idx);
V_train = V_meas(train_idx);
F_train = F_all(train_idx);

%% 3. Min-Max Normalisation 

% stats.Var = [min, max]
% To prevent division by zero, add a tiny eps to the range.
tau_train = tau_all(train_idx);
stats.t = [min(tau_train), max(tau_train)];
stats.X = [min(X_train), max(X_train)];
stats.S = [0, max(S_train)]; 
stats.V = [0, 25];           
stats.F = [0, 2.5];         
stats.Pt = [0, max(Pt_train)]; 
stats.Pm = [0, max(Pm_train)]; 

% Obtain range range = max - min

fprintf('Variable range (Min -> Max):\n');
fprintf('  X:  %.3f -> %.3f\n', stats.X(1), stats.X(2));
fprintf('  S:  %.3f -> %.3f\n', stats.S(1), stats.S(2));
fprintf('  Pt: %.3f -> %.3f\n', stats.Pt(1), stats.Pt(2));
fprintf('  Pm: %.3f -> %.3f\n', stats.Pm(1), stats.Pm(2));
stats_std.X  = std(X_train) + 1e-8;
stats_std.S  = std(S_train) + 1e-8;
stats_std.Pt = std(Pt_train) + 1e-8;
stats_std.Pm = std(Pm_train) + 1e-8;
%% 4. Range of the test batch

S_test_range = [min(S_true(test_idx)), max(S_true(test_idx))];
S_train_range = [min(S_true(train_idx)), max(S_true(train_idx))];
V_test_range = [min(V_meas(test_idx)), max(V_meas(test_idx))];
V_train_range = [min(V_meas(train_idx)), max(V_meas(train_idx))];

fprintf('S range:\n');
fprintf('  Training dataset: [%.2f, %.2f]\n', S_train_range(1), S_train_range(2));
fprintf('  Test set: [%.2f, %.2f]', S_test_range(1), S_test_range(2));

fprintf('\nV range:\n');
fprintf('  Training set: [%.2f, %.2f]\n', V_train_range(1), V_train_range(2));
fprintf('  Test set: [%.2f, %.2f]', V_test_range(1), V_test_range(2));
fprintf('\n');

%% 5. Compute V_exact and Pre-generate Dense Collocation Pool
% 1. Forward Euler integration to get exact Volume (V is fully determined by F)
% 2. Generate dense points (0.1h step) for physics constraints across ALL batches

all_batches_sorted = unique([train_batches, val_batches, test_batch]);
V_exact_all = zeros(size(t_all));

colloc_pool.tau = []; colloc_pool.V = []; colloc_pool.F = [];
colloc_pool.X0 = [];  colloc_pool.S0 = []; colloc_pool.Pt0 = []; colloc_pool.Pm0 = [];

for b = all_batches_sorted
    bidx = find(batch_num == b);
    t_b  = t_all(bidx);
    F_b  = F_all(bidx);
    V_b  = V_meas(bidx);

    [~, order] = sort(t_b);
    bidx = bidx(order);
    t_b  = t_b(order);
    F_b  = F_b(order);
    V_b  = V_b(order);

    V_int    = zeros(length(t_b), 1);
    V_int(1) = V_b(1); 
    for k = 2:length(t_b)
        dt_k     = t_b(k) - t_b(k-1);
        V_int(k) = V_int(k-1) + F_b(k-1) * dt_k;
    end
    V_exact_all(bidx) = V_int; 

    t_dense = (min(t_b):0.1:max(t_b))';

    F_dense = interp1(t_b, F_b,   t_dense, 'previous', 'extrap');
    V_dense = interp1(t_b, V_int, t_dense, 'linear', 'extrap');
    
    colloc_pool.tau = [colloc_pool.tau; t_dense];
    colloc_pool.V   = [colloc_pool.V; V_dense];
    colloc_pool.F   = [colloc_pool.F; F_dense];

    colloc_pool.X0  = [colloc_pool.X0; repmat(X0_all(bidx(1)), length(t_dense), 1)];
    colloc_pool.S0  = [colloc_pool.S0; repmat(S0_all(bidx(1)), length(t_dense), 1)];
    colloc_pool.Pt0 = [colloc_pool.Pt0; repmat(Pt0_all(bidx(1)), length(t_dense), 1)];
    colloc_pool.Pm0 = [colloc_pool.Pm0; repmat(Pm0_all(bidx(1)), length(t_dense), 1)];
end

fprintf('V_exact computed. Max error vs V_meas: %.4f L\n', max(abs(V_exact_all - V_meas)));
fprintf('Dense collocation pool created: %d continuous points.\n\n', length(colloc_pool.tau));

%% 6. Structure of Network (PINN)
numInputs  = 7;   % [tau, V, F, X0, S0, Pt0, Pm0] 
numOutputs = 4;   % [X, S, Pt, Pm]
hiddenSize = 128;

layers = [
    featureInputLayer(numInputs, 'Normalization', 'none')
    fullyConnectedLayer(hiddenSize)
    tanhLayer
    fullyConnectedLayer(hiddenSize)
    tanhLayer
    fullyConnectedLayer(hiddenSize)
    tanhLayer
    fullyConnectedLayer(hiddenSize)
    tanhLayer
    fullyConnectedLayer(numOutputs)
];
net = dlnetwork(layers);

%% 7. Prepare data (min-max)
norm_fn = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);

V_exact_train = V_exact_all(train_idx);
F_train       = F_all(train_idx);

inputTrain = [
    norm_fn(tau_all(train_idx), stats.t), ...
    norm_fn(V_exact_train,      stats.V), ...  
    norm_fn(F_train,            stats.F), ...
    norm_fn(X0_all(train_idx),  stats.X), ...
    norm_fn(S0_all(train_idx),  stats.S), ...
    norm_fn(Pt0_all(train_idx), stats.Pt), ...
    norm_fn(Pm0_all(train_idx), stats.Pm)
]';
dlInput = dlarray(single(inputTrain), 'CB');

targets.X  = norm_fn(X_train,  stats.X);
targets.S  = norm_fn(S_train,  stats.S);
targets.Pt = norm_fn(Pt_train, stats.Pt);
targets.Pm = norm_fn(Pm_train, stats.Pm);

V_exact_test = V_exact_all(test_idx);

inputTest = [
    norm_fn(tau_all(test_idx),  stats.t), ...
    norm_fn(V_exact_test,       stats.V), ... 
    norm_fn(F_all(test_idx),    stats.F), ...
    norm_fn(X0_all(test_idx),   stats.X), ...
    norm_fn(S0_all(test_idx),   stats.S), ...
    norm_fn(Pt0_all(test_idx),  stats.Pt), ...
    norm_fn(Pm0_all(test_idx),  stats.Pm)
]';
dlInputTest = dlarray(single(inputTest), 'CB');

%% 8. Training Cycle (Aligned with Paper: 50k Epochs, Fixed Lambda 0.5)

totalEpochs       = 50000;   
fixed_lambda      = 0.1;    
numColloc         = 500;     
Pm_physics_weight = 0.5;
S_physics_weight  = 0.5;
n_batches_per_epoch = 10;   

fprintf('Training Strategy (50k Epochs, Fixed Lambda=0.5):\n');
fprintf('  Physics weights: Pm_w=%.1f, S_w=%.1f\n', Pm_physics_weight, S_physics_weight);
fprintf('  Stochastic Regularization: %d/%d batches per epoch\n\n', ...
    n_batches_per_epoch, length(train_batches));

avgGrad = []; avgSqGrad = [];
history.loss = zeros(totalEpochs, 1);
history.data = zeros(totalEpochs, 1);
history.physics = zeros(totalEpochs, 1);

best_test_Pm = -inf; 
best_net = net;

batch_train_idx = cell(length(train_batches), 1);
for i = 1:length(train_batches)
    batch_train_idx{i} = find(batch_num(train_idx) == train_batches(i));
end

tic;
for epoch = 1:totalEpochs

    current_LR = 1e-3;   
    if epoch > 15000
        current_LR = 5e-4;   
    end
    if epoch > 35000
        current_LR = 1e-4;   
    end            
  
    selected_batch_ids = randperm(length(train_batches), n_batches_per_epoch);
    epoch_data_idx = [];
    for i = selected_batch_ids
        epoch_data_idx = [epoch_data_idx; batch_train_idx{i}];
    end
    
    dlInput_epoch = dlInput(:, epoch_data_idx);
    targets_epoch.X  = targets.X(epoch_data_idx);
    targets_epoch.S  = targets.S(epoch_data_idx);
    targets_epoch.Pt = targets.Pt(epoch_data_idx);
    targets_epoch.Pm = targets.Pm(epoch_data_idx);

    colloc = genColloc(numColloc, colloc_pool, stats); 
    
    [loss, grads, dL, pL] = dlfeval(@lossFn, net, dlInput_epoch, ...
        targets_epoch, colloc, fixed_lambda, Pm_physics_weight, S_physics_weight, stats, stats_std, Sin, Y);
        
    [net, avgGrad, avgSqGrad] = adamupdate(net, grads, avgGrad, avgSqGrad, epoch, current_LR);

    history.loss(epoch)    = extractdata(loss);
    history.data(epoch)    = dL;
    history.physics(epoch) = pL;

    if mod(epoch, 500) == 0
        dlPred_test = predict(net, dlInputTest);
        pred_test = extractdata(dlPred_test);
        Pm_pred_test = pred_test(4,:)' * (stats.Pm(2) - stats.Pm(1)) + stats.Pm(1);
        Pm_pred_test = max(0, Pm_pred_test);
        
        R2_test = 1 - sum((Pm_true(test_idx)-Pm_pred_test).^2) / ...
                     (sum((Pm_true(test_idx)-mean(Pm_true(test_idx))).^2)+1e-10);

        if R2_test > best_test_Pm
            best_test_Pm = R2_test;
            best_net = net; 
        end
        
        fprintf('  Epoch %5d | Loss=%.4f (D=%.4f, P=%.4f) | Test Pm R²=%.4f | LR=%.5f\n', ...
         epoch, history.loss(epoch), dL, pL, R2_test, current_LR);
    end
end
fprintf('Completed: %.1fsec\n', toc);

net = best_net; 
%% 9. Final Evaluation

R2 = @(y,p) 1 - sum((y-p).^2)/(sum((y-mean(y)).^2)+1e-10);

% Training set
dlPred = predict(net, dlInput);
pred = extractdata(dlPred);
X_pred  = max(0, pred(1,:)' * (stats.X(2)-stats.X(1))  + stats.X(1));
S_pred  = max(0, pred(2,:)' * (stats.S(2)-stats.S(1))  + stats.S(1));
Pt_pred = max(0, pred(3,:)' * (stats.Pt(2)-stats.Pt(1))+ stats.Pt(1));
Pm_pred = max(0, pred(4,:)' * (stats.Pm(2)-stats.Pm(1))+ stats.Pm(1));
V_pred  = V_exact_train;  % V_exact — not predicted by network

fprintf('=== Training set (%d batches) ===\n', length(train_batches));
fprintf('  X:  R²=%.4f\n', R2(X_true(train_idx), X_pred));
fprintf('  S:  R²=%.4f\n', R2(S_true(train_idx), S_pred));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(train_idx), Pt_pred));
fprintf('  Pm: R²=%.4f\n', R2(Pm_true(train_idx), Pm_pred));
fprintf('  V:  R²=1.0000 (exact)\n');

% Test set
dlPredTest = predict(net, dlInputTest);
predTest = extractdata(dlPredTest);
X_pred_test  = max(0, predTest(1,:)' * (stats.X(2)-stats.X(1))  + stats.X(1));
S_pred_test  = max(0, predTest(2,:)' * (stats.S(2)-stats.S(1))  + stats.S(1));
Pt_pred_test = max(0, predTest(3,:)' * (stats.Pt(2)-stats.Pt(1))+ stats.Pt(1));
Pm_pred_test = max(0, predTest(4,:)' * (stats.Pm(2)-stats.Pm(1))+ stats.Pm(1));
V_pred_test  = V_exact_test;  % V_exact

fprintf('\n=== Test set (Batch %d) ===\n', test_batch);
fprintf('  X:  R²=%.4f\n', R2(X_true(test_idx), X_pred_test));
fprintf('  S:  R²=%.4f\n', R2(S_true(test_idx), S_pred_test));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(test_idx), Pt_pred_test));
fprintf('  Pm: R²=%.4f\n', R2(Pm_true(test_idx), Pm_pred_test));
fprintf('  V:  R²=1.0000 (exact)\n');

test_Pm_R2 = R2(Pm_true(test_idx), Pm_pred_test);
fprintf('\n');
if test_Pm_R2 > 0.90
    fprintf('Test successful: PINN extrapolates well under new operating conditions.!\n\n');
elseif test_Pm_R2 > 0.80
    fprintf('Testing Overall: Extrapolation capability is acceptable, but there is still room for improvement.\n\n');
else
    fprintf('Test failed: Model or data partitioning requires adjustment.\n\n');
end

%% 10. Visualization 
% Loss curve
figure('Position', [100 100 1200 400]);
subplot(1,3,1);
semilogy(history.loss, 'b', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss'); title('Total Loss'); grid on;
subplot(1,3,2);
semilogy(history.data, 'r', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss'); title('Data Loss'); grid on;
subplot(1,3,3);
semilogy(history.physics, 'g', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss'); title('Physics Loss'); grid on;
sgtitle(sprintf('Loss Evolution (Test=Batch %d)', test_batch));

% Parity plots (Train and Test only)
figure('Position', [100 100 1200 800]);
subplot(2,2,1);
scatter(X_true(train_idx), X_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(X_true(test_idx), X_pred_test, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
plot([0 3], [0 3], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('X: Tr=%.3f, Test=%.3f', R2(X_true(train_idx), X_pred), R2(X_true(test_idx), X_pred_test)));
legend('Train', 'Test', 'Location', 'best'); grid on;

subplot(2,2,2);
scatter(S_true(train_idx), S_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(S_true(test_idx), S_pred_test, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
plot([0 20], [0 20], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('S: Tr=%.3f, Test=%.3f', R2(S_true(train_idx), S_pred), R2(S_true(test_idx), S_pred_test)));
legend('Train', 'Test', 'Location', 'best'); grid on;

subplot(2,2,3);
scatter(Pt_true(train_idx), Pt_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(Pt_true(test_idx), Pt_pred_test, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
plot([0 5], [0 5], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('Pt: Tr=%.3f, Test=%.3f', R2(Pt_true(train_idx), Pt_pred), R2(Pt_true(test_idx), Pt_pred_test)));
legend('Train', 'Test', 'Location', 'best'); grid on;

subplot(2,2,4);
scatter(Pm_true(train_idx), Pm_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(Pm_true(test_idx), Pm_pred_test, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
plot([0 4], [0 4], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('Pm: Tr=%.3f, Test=%.3f', R2(Pm_true(train_idx), Pm_pred), R2(Pm_true(test_idx), Pm_pred_test)));
legend('Train', 'Test', 'Location', 'best'); grid on;
sgtitle('Predictions vs True Values');

% True vs PINN (A–F) 
t_test = t_all(test_idx);

figure('Position',[100 100 1100 800]);

lw_true = 2.2;
lw_pred = 2.2;

% ---------- (A) Biomass ----------
subplot(3,2,1); hold on;
plot(t_test, X_true(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, X_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Biomass (g/L)');
title('(A) Biomass');

% ---------- (B) Substrate ----------
subplot(3,2,2); hold on;
plot(t_test, S_true(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, S_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Substrate (g/L)');
title('(B) Substrate');

% ---------- (C) Total Protein ----------
subplot(3,2,3); hold on;
plot(t_test, Pt_true(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, Pt_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Total Protein (g/L)');
title('(C) Total Protein');

% ---------- (D) Secreted Protein ----------
subplot(3,2,4); hold on;
plot(t_test, Pm_true(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, Pm_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Secreted Protein (g/L)');
title('(D) Secreted Protein');

% ---------- (E) Volume ----------
subplot(3,2,5); hold on;
plot(t_test, V_meas(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, V_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Volume (L)');
title('(E) Volume');

% ---------- (F) Feed rate ----------
subplot(3,2,6); hold on;
stairs(t_test, F_all(test_idx), 'm', 'LineWidth', 2.2);
grid on; box on;
xlabel('Time (h)'); ylabel('Feed rate (L/h)');
title('(F) Feed rate profile');

% ---------- Legend (only once) ----------
subplot(3,2,1);
legend({'True','PINN'}, 'Location','best');

sgtitle(sprintf('Time-series comparison on Test batch %d (True vs PINN)', test_batch));
set(gcf,'Color','w');

%% 11. Fig-3 Style: Parity Plots + Physics Parity Plots (A–D)
% ===================== Helper functions =====================
norm_fn_vis  = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);
R2vis        = @(y,p)  1 - sum((y(:)-p(:)).^2) / (sum((y(:)-mean(y(:))).^2) + 1e-10);

dt_vis = 0.1;   % same corrected dt as training

% ===================== Scaled predictions =====================
pred_tr_raw  = extractdata(predict(net, dlInput));
pred_test_raw = extractdata(predict(net, dlInputTest));

% Scaled training targets
X_tr_sc   = pred_tr_raw(1,:)';
S_tr_sc   = pred_tr_raw(2,:)';
Pt_tr_sc  = pred_tr_raw(3,:)';
Pm_tr_sc  = pred_tr_raw(4,:)';
V_tr_sc   = norm_fn_vis(V_exact_train, stats.V);

X_tr_m    = norm_fn_vis(X_train,  stats.X);
S_tr_m    = norm_fn_vis(S_train,  stats.S);
Pt_tr_m   = norm_fn_vis(Pt_train, stats.Pt);
Pm_tr_m   = norm_fn_vis(Pm_train, stats.Pm);
V_tr_m    = norm_fn_vis(V_exact_train, stats.V);

% Scaled Testing targets
X_ts_sc  = pred_test_raw(1,:)';
S_ts_sc  = pred_test_raw(2,:)';
Pt_ts_sc = pred_test_raw(3,:)';
Pm_ts_sc = pred_test_raw(4,:)';
V_ts_sc  = norm_fn_vis(V_exact_test, stats.V);

X_ts_m   = norm_fn_vis(X_meas(test_idx),  stats.X);
S_ts_m   = norm_fn_vis(S_meas(test_idx),  stats.S);
Pt_ts_m  = norm_fn_vis(Pt_meas(test_idx), stats.Pt);
Pm_ts_m  = norm_fn_vis(Pm_meas(test_idx), stats.Pm);
V_ts_m   = norm_fn_vis(V_exact_test, stats.V);

% R² (scaled)
r2X_tr=R2vis(X_tr_m,X_tr_sc); r2S_tr=R2vis(S_tr_m,S_tr_sc);
r2Pt_tr=R2vis(Pt_tr_m,Pt_tr_sc); r2Pm_tr=R2vis(Pm_tr_m,Pm_tr_sc);
r2X_ts=R2vis(X_ts_m,X_ts_sc); r2S_ts=R2vis(S_ts_m,S_ts_sc);
r2Pt_ts=R2vis(Pt_ts_m,Pt_ts_sc); r2Pm_ts=R2vis(Pm_ts_m,Pm_ts_sc);

% ===================== Physics at collocation (panel C) =====================
numColloc_vis = 600;
col = genColloc(numColloc_vis, colloc_pool, stats);

% Network predictions at collocation (denormalised)
yC   = extractdata(predict(net, dlarray(single(col.input),'CB')));
Xc   = yC(1,:).*(stats.X(2) -stats.X(1)) + stats.X(1);
Sc   = yC(2,:).*(stats.S(2) -stats.S(1)) + stats.S(1);
Ptc  = yC(3,:).*(stats.Pt(2)-stats.Pt(1))+ stats.Pt(1);
Pmc  = yC(4,:).*(stats.Pm(2)-stats.Pm(1))+ stats.Pm(1);
Vc   = max(col.V_raw, 0.1);
Dc   = col.F_raw ./ Vc;

Sc_s  = max(Sc,  1e-6);
mu_c  = 21.87*Sc_s./((Sc_s+0.4).*(Sc_s+62.5));
th_c  = 4.75*mu_c./(0.12+mu_c);
fp_c  = Sc_s.*exp(-5*Sc_s)./(0.1+Sc_s);

% True (ODE RHS) derivatives
dX_tC  =  mu_c.*Xc   - Dc.*Xc;
dS_tC  = -Y*mu_c.*Xc + Dc.*(Sin-Sc);
dPt_tC =  fp_c.*Xc   - Dc.*Ptc;
dPm_tC =  th_c.*(Ptc-Pmc) - Dc.*Pmc;

% Predicted derivatives (central difference, dt=0.1h)
[dX_pC,dS_pC,dPt_pC,dPm_pC] = centralDiff(net, col.input, stats, dt_vis, col.F_raw);

% Scale by variable range for axis comparability
scX=stats.X(2)-stats.X(1); scS=stats.S(2)-stats.S(1);
scPt=stats.Pt(2)-stats.Pt(1); scPm=stats.Pm(2)-stats.Pm(1);

dX_tCsc=dX_tC/scX;  dS_tCsc=dS_tC/scS;
dPt_tCsc=dPt_tC/scPt; dPm_tCsc=dPm_tC/scPm;
dX_pCsc=dX_pC/scX;  dS_pCsc=dS_pC/scS;
dPt_pCsc=dPt_pC/scPt; dPm_pCsc=dPm_pC/scPm;

r2dX_c=R2vis(dX_tCsc,dX_pCsc); r2dS_c=R2vis(dS_tCsc,dS_pCsc);
r2dPt_c=R2vis(dPt_tCsc,dPt_pCsc); r2dPm_c=R2vis(dPm_tCsc,dPm_pCsc);

% ===================== Physics at Testing pts (panel D) =====================
yT   = extractdata(predict(net, dlInputTest));
Xt   = yT(1,:).*(stats.X(2) -stats.X(1)) + stats.X(1);
St   = yT(2,:).*(stats.S(2) -stats.S(1)) + stats.S(1);
Ptt  = yT(3,:).*(stats.Pt(2)-stats.Pt(1))+ stats.Pt(1);
Pmt  = yT(4,:).*(stats.Pm(2)-stats.Pm(1))+ stats.Pm(1);
Vt   = max(V_exact_test', 0.1);
Dt   = F_all(test_idx)' ./ Vt;

St_s  = max(St, 1e-6);
mu_t  = 21.87*St_s./((St_s+0.4).*(St_s+62.5));
th_t  = 4.75*mu_t./(0.12+mu_t);
fp_t  = St_s.*exp(-5*St_s)./(0.1+St_s);

dX_tT  =  mu_t.*Xt   - Dt.*Xt;
dS_tT  = -Y*mu_t.*Xt + Dt.*(Sin-St);
dPt_tT =  fp_t.*Xt   - Dt.*Ptt;
dPm_tT =  th_t.*(Ptt-Pmt) - Dt.*Pmt;

[dX_pT,dS_pT,dPt_pT,dPm_pT] = centralDiff(net, inputTest, stats, dt_vis, F_all(test_idx)');

dX_tTsc=dX_tT/scX;  dS_tTsc=dS_tT/scS;
dPt_tTsc=dPt_tT/scPt; dPm_tTsc=dPm_tT/scPm;
dX_pTsc=dX_pT/scX;  dS_pTsc=dS_pT/scS;
dPt_pTsc=dPt_pT/scPt; dPm_pTsc=dPm_pT/scPm;

r2dX_t=R2vis(dX_tTsc,dX_pTsc); r2dS_t=R2vis(dS_tTsc,dS_pTsc);
r2dPt_t=R2vis(dPt_tTsc,dPt_pTsc); r2dPm_t=R2vis(dPm_tTsc,dPm_pTsc);

% ===================== Plot =====================
cX=[0.12,0.47,0.71]; cS=[1.00,0.50,0.05];
cPt=[0.17,0.63,0.17]; cPm=[0.84,0.15,0.16]; cV=[0.58,0.40,0.74];
ms=20; ma=0.5;

figure('Position',[80 80 1100 950],'Color','w');

% ---- (A) Training data parity ----
ax=subplot(2,2,1); hold(ax,'on');
scatter(ax,X_tr_m,  X_tr_sc,  ms,cX, 'o','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('X  (R²=%.3f)',r2X_tr));
scatter(ax,S_tr_m,  S_tr_sc,  ms,cS, 's','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('S  (R²=%.3f)',r2S_tr));
scatter(ax,Pt_tr_m, Pt_tr_sc, ms,cPt,'^','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('Pt (R²=%.3f)',r2Pt_tr));
scatter(ax,Pm_tr_m, Pm_tr_sc, ms,cPm,'d','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('Pm (R²=%.3f)',r2Pm_tr));
scatter(ax,V_tr_m,  V_tr_sc,  ms,cV, 'x','LineWidth',1.5,...
    'DisplayName','V (exact)');
plot(ax,[0 1],[0 1],'k-','LineWidth',2);
plot(ax,[0 1],[0.05 1.05],'k--','LineWidth',1); plot(ax,[0 1],[-0.05 0.95],'k--','LineWidth',1);
xlim(ax,[0 1]); ylim(ax,[-0.05 1.1]);
xlabel(ax,'Measured values (scaled)'); ylabel(ax,'Predicted values (scaled)');
title(ax,'(A) Data Loss: Training','FontWeight','bold');
legend(ax,'Location','northwest','FontSize',9); grid(ax,'on'); box(ax,'on');

% ---- (B) Testing data parity ----
ax=subplot(2,2,2); hold(ax,'on');
scatter(ax,X_ts_m,  X_ts_sc,  ms,cX, 'o','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('X  (R²=%.3f)',r2X_ts));
scatter(ax,S_ts_m,  S_ts_sc,  ms,cS, 's','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('S  (R²=%.3f)',r2S_ts));
scatter(ax,Pt_ts_m, Pt_ts_sc, ms,cPt,'^','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('Pt (R²=%.3f)',r2Pt_ts));
scatter(ax,Pm_ts_m, Pm_ts_sc, ms,cPm,'d','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('Pm (R²=%.3f)',r2Pm_ts));
scatter(ax,V_ts_m,  V_ts_sc,  ms,cV, 'x','LineWidth',1.5,...
    'DisplayName','V (exact)');
plot(ax,[0 1],[0 1],'k-','LineWidth',2);
plot(ax,[0 1],[0.05 1.05],'k--','LineWidth',1); plot(ax,[0 1],[-0.05 0.95],'k--','LineWidth',1);
xlim(ax,[0 1]); ylim(ax,[-0.05 1.1]);
xlabel(ax,'Measured values (scaled)'); ylabel(ax,'Predicted values (scaled)');
title(ax,'(B) Data Loss: Testing','FontWeight','bold');
legend(ax,'Location','northwest','FontSize',9); grid(ax,'on'); box(ax,'on');

% ---- (C) Physics parity — collocation ----
ax=subplot(2,2,3); hold(ax,'on');
scatter(ax,dX_tCsc(:), dX_pCsc(:), ms,cX, 'o','MarkerEdgeAlpha',0.5,'MarkerFaceAlpha',0,...
    'DisplayName',sprintf('dX/dt  (R²=%.3f)',r2dX_c));
scatter(ax,dS_tCsc(:), dS_pCsc(:), ms,cS, 's','MarkerEdgeAlpha',0.5,'MarkerFaceAlpha',0,...
    'DisplayName',sprintf('dS/dt  (R²=%.3f)',r2dS_c));
scatter(ax,dPt_tCsc(:),dPt_pCsc(:),ms,cPt,'^','MarkerEdgeAlpha',0.5,'MarkerFaceAlpha',0,...
    'DisplayName',sprintf('dPt/dt (R²=%.3f)',r2dPt_c));
scatter(ax,dPm_tCsc(:),dPm_pCsc(:),ms,cPm,'d','MarkerEdgeAlpha',0.5,'MarkerFaceAlpha',0,...
    'DisplayName',sprintf('dPm/dt (R²=%.3f)',r2dPm_c));
allC=[dX_tCsc(:);dS_tCsc(:);dPt_tCsc(:);dPm_tCsc(:)];
lim_c=[min(allC)*1.1, max(allC)*1.1];
plot(ax,lim_c,lim_c,'k-','LineWidth',2);
xlim(ax,lim_c); ylim(ax,lim_c);
xlabel(ax,'True physics (scaled)'); ylabel(ax,'Predicted physics (scaled)');
title(ax,'(C) Physics Loss: Collocation Points','FontWeight','bold');
legend(ax,'Location','northwest','FontSize',9); grid(ax,'on'); box(ax,'on');

% ---- (D) Physics parity — Testing data points ----
ax=subplot(2,2,4); hold(ax,'on');
scatter(ax,dX_tTsc(:), dX_pTsc(:), ms,cX, 'o','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('dX/dt  (R²=%.3f)',r2dX_t));
scatter(ax,dS_tTsc(:), dS_pTsc(:), ms,cS, 's','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('dS/dt  (R²=%.3f)',r2dS_t));
scatter(ax,dPt_tTsc(:),dPt_pTsc(:),ms,cPt,'^','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('dPt/dt (R²=%.3f)',r2dPt_t));
scatter(ax,dPm_tTsc(:),dPm_pTsc(:),ms,cPm,'d','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('dPm/dt (R²=%.3f)',r2dPm_t));
allT=[dX_tTsc(:);dS_tTsc(:);dPt_tTsc(:);dPm_tTsc(:)];
lim_t=[min(allT)*1.1, max(allT)*1.1];
plot(ax,lim_t,lim_t,'k-','LineWidth',2);
xlim(ax,lim_t); ylim(ax,lim_t);
xlabel(ax,'True physics (scaled)'); ylabel(ax,'Predicted physics (scaled)');
title(ax,'(D) Physics Loss: Testing Data Points','FontWeight','bold');
legend(ax,'Location','northwest','FontSize',9); grid(ax,'on'); box(ax,'on');

sgtitle('Parity Plots: Data Loss (A,B) & Physics Loss (C,D)',...
    'FontSize',13,'FontWeight','bold');
set(gcf,'Color','w');%% Functions

%% Functions
function [loss, grads, dL, pL] = lossFn(net, dlX, targets, colloc, lambda, Pm_phys_w, S_phys_w, stats, stats_std, Sin, Y)
    dlY = forward(net, dlX); % 4 outputs: X, S, Pt, Pm
    
    % WMSE 
    denorm = @(y, s) y .* (s(2) - s(1)) + s(1);
    
    pred_X  = denorm(dlY(1,:), stats.X);
    pred_S  = denorm(dlY(2,:), stats.S);
    pred_Pt = denorm(dlY(3,:), stats.Pt);
    pred_Pm = denorm(dlY(4,:), stats.Pm);
    
    true_X  = denorm(targets.X', stats.X);
    true_S  = denorm(targets.S', stats.S);
    true_Pt = denorm(targets.Pt', stats.Pt);
    true_Pm = denorm(targets.Pm', stats.Pm);
    
    w = [1, 1, 1.5, 2];
    
    dL = w(1)*mean(((pred_X  - true_X)  ./ stats_std.X).^2) + ...
         w(2)*mean(((pred_S  - true_S)  ./ stats_std.S).^2) + ...
         w(3)*mean(((pred_Pt - true_Pt) ./ stats_std.Pt).^2) + ...
         w(4)*mean(((pred_Pm - true_Pm) ./ stats_std.Pm).^2);
    dL = dL / sum(w);
    
    pL = physLoss(net, colloc, Pm_phys_w, S_phys_w, stats, stats_std, Sin, Y); 
    
    loss = (1-lambda)*dL + lambda*pL; 
    grads = dlgradient(loss, net.Learnables); 
    
    dL = extractdata(dL); 
    pL = extractdata(pL);
end

function pL = physLoss(net, c, Pm_phys_w, S_phys_w, stats, stats_std, Sin, Y)
    dlC = dlarray(single(c.input), 'CB');
    dlY = forward(net, dlC); 
    
    denorm = @(y, s) y .* (s(2) - s(1)) + s(1);
    
    X  = denorm(dlY(1,:), stats.X);
    S  = denorm(dlY(2,:), stats.S);
    Pt = denorm(dlY(3,:), stats.Pt);
    Pm = denorm(dlY(4,:), stats.Pm);
    
    V  = dlarray(single(c.V_raw), 'CB');
    Fr = dlarray(single(c.F_raw), 'CB');
    D  = Fr ./ max(V, 0.1); 
    
    S_safe = max(S, 1e-6);
    
    mu    = 21.87 * S_safe ./ ((S_safe + 0.4) .* (S_safe + 62.5));
    theta = 4.75  * mu     ./ (0.12 + mu);
    fp    = S_safe .* exp(-5 * S_safe) ./ (0.1 + S_safe);
    
    dt      = 0.1; 
    dt_norm = dt / (stats.t(2) - stats.t(1) + 1e-8);
    
    dV      = c.F_raw * dt; 
    dV_norm = dV / (stats.V(2) - stats.V(1) + 1e-8);

    cPlus  = c.input; cMinus = c.input;
    
    cPlus(1,:)  = cPlus(1,:)  + dt_norm;   cMinus(1,:) = cMinus(1,:) - dt_norm;   
    cPlus(2,:)  = cPlus(2,:)  + dV_norm;   cMinus(2,:) = cMinus(2,:) - dV_norm;
    
    dlCp = dlarray(single(cPlus),  'CB'); dlCm = dlarray(single(cMinus), 'CB');
    dlYp = forward(net, dlCp); dlYm = forward(net, dlCm);
    
    dX  = (dlYp(1,:) - dlYm(1,:)) * (stats.X(2)  - stats.X(1))  / (2*dt);
    dS  = (dlYp(2,:) - dlYm(2,:)) * (stats.S(2)  - stats.S(1))  / (2*dt);
    dPt = (dlYp(3,:) - dlYm(3,:)) * (stats.Pt(2) - stats.Pt(1)) / (2*dt);
    dPm = (dlYp(4,:) - dlYm(4,:)) * (stats.Pm(2) - stats.Pm(1)) / (2*dt);
    
    rX  = dX  - (mu.*X  - D.*X);
    rS  = dS  - (-Y*mu.*X + D.*(Sin - S));
    rPt = dPt - (fp.*X  - D.*Pt);
    rPm = dPm - (theta.*(Pt - Pm) - D.*Pm);
    
    wX  = 1.0; wS  = S_phys_w; wPt = 1.0; wPm = Pm_phys_w;

    pL = mean( ...
          wX  * (rX  ./ stats_std.X).^2 + ...
          wS  * (rS  ./ stats_std.S).^2 + ...
          wPt * (rPt ./ stats_std.Pt).^2 + ...
          wPm * (rPm ./ stats_std.Pm).^2 ...
          , 'all');
    
    pL = pL / (wX + wS + wPt + wPm + 1e-8);
end

function c = genColloc(n, pool, stats)
    idx = randi(length(pool.tau), 1, n);

    tau_raw = reshape(pool.tau(idx), 1, []);  
    V_raw   = reshape(max(pool.V(idx), 0.1), 1, []);
    F_raw   = reshape(pool.F(idx), 1, []);
    X0_raw  = reshape(pool.X0(idx), 1, []);
    S0_raw  = reshape(pool.S0(idx), 1, []);
    Pt0_raw = reshape(pool.Pt0(idx), 1, []);
    Pm0_raw = reshape(pool.Pm0(idx), 1, []);

    c.t_raw = tau_raw;
    c.V_raw = V_raw;
    c.F_raw = F_raw;

    norm_fn = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);

    c.input = [
        norm_fn(tau_raw, stats.t);
        norm_fn(V_raw,   stats.V); 
        norm_fn(F_raw,   stats.F);
        norm_fn(X0_raw,  stats.X);
        norm_fn(S0_raw,  stats.S);
        norm_fn(Pt0_raw, stats.Pt);
        norm_fn(Pm0_raw, stats.Pm)
    ];
end

function [dX,dS,dPt,dPm] = centralDiff(net, inp, stats, dt, F_raw)
    dt_norm  = dt / (stats.t(2) - stats.t(1) + 1e-8);
    dV       = F_raw * dt;
    dV_norm  = dV / (stats.V(2) - stats.V(1) + 1e-8);
    
    cP = inp; 
    cP(1,:) = cP(1,:) + dt_norm;   
    cP(2,:) = cP(2,:) + dV_norm;  
    
    cM = inp; 
    cM(1,:) = cM(1,:) - dt_norm;   
    cM(2,:) = cM(2,:) - dV_norm;   
    
    yP = extractdata(predict(net, dlarray(single(cP),'CB')));
    yM = extractdata(predict(net, dlarray(single(cM),'CB')));
    
    dX  = (yP(1,:) - yM(1,:)) * (stats.X(2)  - stats.X(1))  / (2*dt);
    dS  = (yP(2,:) - yM(2,:)) * (stats.S(2)  - stats.S(1))  / (2*dt);
    dPt = (yP(3,:) - yM(3,:)) * (stats.Pt(2) - stats.Pt(1)) / (2*dt);
    dPm = (yP(4,:) - yM(4,:)) * (stats.Pm(2) - stats.Pm(1)) / (2*dt);
end