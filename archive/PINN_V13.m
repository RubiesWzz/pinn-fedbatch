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
train_batches_all = [1, 3, 4, 7, 8, 11, 12, 13, 14, 15];
train_batches = setdiff(train_batches_all, test_batch);

% Validation set
val_batches = [2, 5, 6, 9, 10];

train_idx = ismember(batch_num, train_batches);
val_idx = ismember(batch_num, val_batches);
test_idx = (batch_num == test_batch);

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
stats.t = [min(t_train), max(t_train)];
stats.X = [min(X_train), max(X_train)];
stats.S = [0, max(S_train)]; 
stats.V = [0, 25];           
stats.F = [0, 2.5];         
stats.Pt = [0, max(Pt_train)]; 
stats.Pm = [0, max(Pm_train)]; 

% Obtain range range = max - min
get_range = @(s) max(s(2) - s(1), 1e-6);

fprintf('Variable range (Min -> Max):\n');
fprintf('  X:  %.3f -> %.3f\n', stats.X(1), stats.X(2));
fprintf('  S:  %.3f -> %.3f\n', stats.S(1), stats.S(2));
fprintf('  Pt: %.3f -> %.3f\n', stats.Pt(1), stats.Pt(2));
fprintf('  Pm: %.3f -> %.3f\n', stats.Pm(1), stats.Pm(2));

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

%% 5. Compute V_exact via numerical integration of dV/dt = F
% V is fully determined by F(t) — no need for the network to predict it.
% Forward Euler integration per batch from V0 = V_meas(t=0).
all_batches_sorted = unique([train_batches, val_batches, test_batch]);
V_exact_all = zeros(size(t_all));

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
    V_int(1) = V_b(1);  % initial condition from measurement
    for k = 2:length(t_b)
        dt_k     = t_b(k) - t_b(k-1);
        V_int(k) = V_int(k-1) + F_b(k-1) * dt_k;
    end
    V_exact_all(bidx) = V_int;
end

fprintf('V_exact computed. Max error vs V_meas: %.4f L\n\n', ...
    max(abs(V_exact_all - V_meas)));

%% 6. Structure of Network
numInputs  = 7;   % inputs unchanged: [t, X, S, Pt, Pm, V_exact, F]
numOutputs = 4;   % X, S, Pt, Pm  (V removed — exact)
hiddenSize = 50;

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
    sigmoidLayer
];
net = dlnetwork(layers);
fprintf('Network Parameters: %d\n\n', sum(cellfun(@numel, net.Learnables.Value)));

%% 6. Training Strategy  (FIXED lambda)
lambda_fixed = 0.5;     % <<< 固定为 0.5

stage1_epochs = 2000;
stage2_epochs = 3000;
stage3_epochs = 2000;

Pm_physics_weight_s1 = 0.5; 
S_physics_weight_s1  = 1.0;

Pm_physics_weight_s2 = 1.5;
S_physics_weight_s2  = 1.5;

Pm_physics_weight_s3 = 2.0;
S_physics_weight_s3  = 2.0;

totalEpochs  = stage1_epochs + stage2_epochs + stage3_epochs;
learningRate = 0.001;
numColloc    = 350;

n_batches_per_epoch = round(length(train_batches) * 0.67);

fprintf('Training Strategy (Fixed lambda):\n');
fprintf('  lambda_fixed = %.2f\n', lambda_fixed);
fprintf('  Phase 1: Pm_w=%.1f, S_w=%.1f\n', Pm_physics_weight_s1, S_physics_weight_s1);
fprintf('  Phase 2: Pm_w=%.1f, S_w=%.1f\n', Pm_physics_weight_s2, S_physics_weight_s2);
fprintf('  Phase 3: Pm_w=%.1f, S_w=%.1f\n', Pm_physics_weight_s3, S_physics_weight_s3);
fprintf('\n Stochastic Regularization: %d/%d batches per epoch\n\n', ...
    n_batches_per_epoch, length(train_batches));

%% 7. Prepare data (min-max)
norm_fn = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);

% Use V_exact (not V_meas) as row 6 of input — exact volume from dV/dt=F
V_exact_train = V_exact_all(train_idx);
V_exact_val   = V_exact_all(val_idx);
V_exact_test  = V_exact_all(test_idx);

inputTrain = [
    norm_fn(t_train,          stats.t),  ...
    norm_fn(X_train,          stats.X),  ...
    norm_fn(S_train,          stats.S),  ...
    norm_fn(Pt_train,         stats.Pt), ...
    norm_fn(Pm_train,         stats.Pm), ...
    norm_fn(V_exact_train,    stats.V),  ...  % V_exact replaces V_meas
    norm_fn(F_train,          stats.F)
]';
dlInput = dlarray(single(inputTrain), 'CB');

% No targets.V — network does not predict V (v12-exactV: V is exact)
targets.X  = norm_fn(X_train,  stats.X);
targets.S  = norm_fn(S_train,  stats.S);
targets.Pt = norm_fn(Pt_train, stats.Pt);
targets.Pm = norm_fn(Pm_train, stats.Pm);

inputVal = [
    norm_fn(t_all(val_idx),       stats.t),  ...
    norm_fn(X_meas(val_idx),      stats.X),  ...
    norm_fn(S_meas(val_idx),      stats.S),  ...
    norm_fn(Pt_meas(val_idx),     stats.Pt), ...
    norm_fn(Pm_meas(val_idx),     stats.Pm), ...
    norm_fn(V_exact_val,          stats.V),  ...  % V_exact
    norm_fn(F_all(val_idx),       stats.F)
]';
dlInputVal = dlarray(single(inputVal), 'CB');

inputTest = [
    norm_fn(t_all(test_idx),      stats.t),  ...
    norm_fn(X_meas(test_idx),     stats.X),  ...
    norm_fn(S_meas(test_idx),     stats.S),  ...
    norm_fn(Pt_meas(test_idx),    stats.Pt), ...
    norm_fn(Pm_meas(test_idx),    stats.Pm), ...
    norm_fn(V_exact_test,         stats.V),  ...  % V_exact
    norm_fn(F_all(test_idx),      stats.F)
]';
dlInputTest = dlarray(single(inputTest), 'CB');

%% 8. Training Cycle
avgGrad = []; avgSqGrad = [];
history.loss = zeros(totalEpochs, 1);
history.data = zeros(totalEpochs, 1);
history.physics = zeros(totalEpochs, 1);
history.lambda = zeros(totalEpochs, 1);
history.Pm_train = zeros(totalEpochs, 1);
history.Pm_val = zeros(totalEpochs, 1);

best_val_Pm = -inf;
best_net = net;
patience = 0;
max_patience = 3000;

batch_train_idx = cell(length(train_batches), 1);
for i = 1:length(train_batches)
    batch_train_idx{i} = find(batch_num(train_idx) == train_batches(i));
end

fprintf('In training...\n');
tic;

for epoch = 1:totalEpochs

    lambda = lambda_fixed;                
    history.lambda(epoch) = lambda;

    if epoch <= stage1_epochs
        Pm_physics_weight = Pm_physics_weight_s1;
        S_physics_weight  = S_physics_weight_s1;
        stage_name = 'S1';
    elseif epoch <= stage1_epochs + stage2_epochs
        Pm_physics_weight = Pm_physics_weight_s2;
        S_physics_weight  = S_physics_weight_s2;
        stage_name = 'S2';
    else
        Pm_physics_weight = Pm_physics_weight_s3;
        S_physics_weight  = S_physics_weight_s3;
        stage_name = 'S3';
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

    colloc = genColloc(numColloc, t_train, X_train, S_train, ...
        Pt_train, Pm_train, V_train, F_train, stats);

    [loss, grads, dL, pL] = dlfeval(@lossFn, net, dlInput_epoch, ...
        targets_epoch, colloc, lambda, Pm_physics_weight, S_physics_weight, stats, Sin, Y);

    [net, avgGrad, avgSqGrad] = adamupdate(net, grads, avgGrad, ...
        avgSqGrad, epoch, learningRate);

    history.loss(epoch)    = extractdata(loss);
    history.data(epoch)    = dL;
    history.physics(epoch) = pL;
    
    if mod(epoch, 100) == 0
        dlPred_tr = predict(net, dlInput);
        pred_tr = extractdata(dlPred_tr);
        Pm_pred_tr = pred_tr(4,:)' * stats.Pm(2) + stats.Pm(1);
        Pm_pred_tr = max(0, Pm_pred_tr);
        
        R2_train = 1 - sum((Pm_true(train_idx)-Pm_pred_tr).^2) / ...
                       (sum((Pm_true(train_idx)-mean(Pm_true(train_idx))).^2)+1e-10);
        
        dlPred_val = predict(net, dlInputVal);
        pred_val = extractdata(dlPred_val);
        Pm_pred_val = pred_val(4,:)' * stats.Pm(2) + stats.Pm(1);
        Pm_pred_val = max(0, Pm_pred_val);
        
        R2_val = 1 - sum((Pm_true(val_idx)-Pm_pred_val).^2) / ...
                     (sum((Pm_true(val_idx)-mean(Pm_true(val_idx))).^2)+1e-10);
        
        history.Pm_train(epoch) = R2_train;
        history.Pm_val(epoch) = R2_val;
        
        if R2_val > best_val_Pm
            best_val_Pm = R2_val;
            best_net = net;
            patience = 0;
        else
            patience = patience + 100;
        end
        
        if mod(epoch, 500) == 0
            fprintf('  [%s] Epoch %4d | Loss=%.4f (D=%.4f, P=%.4f) | Pm: Tr=%.3f, Val=%.3f', ...
                stage_name, epoch, history.loss(epoch), dL, pL, R2_train, R2_val);
            if R2_val == best_val_Pm
                fprintf(' \n');
            else
                fprintf('\n');
            end
        end
    end
    
    if patience >= max_patience
        fprintf('\nTerminated early at Epoch %d (Best PM Verification R²=%.4f)\n', epoch, best_val_Pm);
        break;
    end
end
fprintf('Completed: %.1fsec\n', toc);
fprintf('Best PM Verification R²: %.4f\n\n', best_val_Pm);

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

% Validation set
dlPredVal = predict(net, dlInputVal);
predVal = extractdata(dlPredVal);
X_pred_val  = max(0, predVal(1,:)' * (stats.X(2)-stats.X(1))  + stats.X(1));
S_pred_val  = max(0, predVal(2,:)' * (stats.S(2)-stats.S(1))  + stats.S(1));
Pt_pred_val = max(0, predVal(3,:)' * (stats.Pt(2)-stats.Pt(1))+ stats.Pt(1));
Pm_pred_val = max(0, predVal(4,:)' * (stats.Pm(2)-stats.Pm(1))+ stats.Pm(1));
V_pred_val  = V_exact_val;  % V_exact

fprintf('\n=== Validation set (%d batches) ===\n', length(val_batches));
fprintf('  X:  R²=%.4f\n', R2(X_true(val_idx), X_pred_val));
fprintf('  S:  R²=%.4f\n', R2(S_true(val_idx), S_pred_val));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(val_idx), Pt_pred_val));
fprintf('  Pm: R²=%.4f\n', R2(Pm_true(val_idx), Pm_pred_val));
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
sgtitle(sprintf('Loss Evolution (Fixed, Test=Batch %d)', test_batch));

% Parity plots
figure('Position', [100 100 1200 800]);
subplot(2,2,1);
scatter(X_true(train_idx), X_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(X_true(val_idx), X_pred_val, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(X_true(test_idx), X_pred_test, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
plot([0 3], [0 3], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('X: Tr=%.3f, Val=%.3f, Test=%.3f', ...
    R2(X_true(train_idx), X_pred), R2(X_true(val_idx), X_pred_val), R2(X_true(test_idx), X_pred_test)));
legend('Train', 'Val', 'Test', 'Location', 'best'); grid on;

subplot(2,2,2);
scatter(S_true(train_idx), S_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(S_true(val_idx), S_pred_val, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(S_true(test_idx), S_pred_test, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
plot([0 20], [0 20], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('S: Tr=%.3f, Val=%.3f, Test=%.3f', ...
    R2(S_true(train_idx), S_pred), R2(S_true(val_idx), S_pred_val), R2(S_true(test_idx), S_pred_test)));
legend('Train', 'Val', 'Test', 'Location', 'best'); grid on;

subplot(2,2,3);
scatter(Pt_true(train_idx), Pt_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(Pt_true(val_idx), Pt_pred_val, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(Pt_true(test_idx), Pt_pred_test, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
plot([0 5], [0 5], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('Pt: Tr=%.3f, Val=%.3f, Test=%.3f', ...
    R2(Pt_true(train_idx), Pt_pred), R2(Pt_true(val_idx), Pt_pred_val), R2(Pt_true(test_idx), Pt_pred_test)));
legend('Train', 'Val', 'Test', 'Location', 'best'); grid on;

subplot(2,2,4);
scatter(Pm_true(train_idx), Pm_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(Pm_true(val_idx), Pm_pred_val, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
scatter(Pm_true(test_idx), Pm_pred_test, 40, 'g', 'filled', 'MarkerFaceAlpha', 0.8);
plot([0 4], [0 4], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('Pm: Tr=%.3f, Val=%.3f, Test=%.3f', ...
    R2(Pm_true(train_idx), Pm_pred), R2(Pm_true(val_idx), Pm_pred_val), R2(Pm_true(test_idx), Pm_pred_test)));
legend('Train', 'Val', 'Test', 'Location', 'best'); grid on;
sgtitle('Predictions vs True Values (Fixed)');

% True vs PINN (A–F) 
t_test = t_all(test_idx);

figure('Position',[100 100 1100 800]);

lw_true = 2.2;
lw_pred = 2.2;
fs = 11;

% ---------- (A) Biomass ----------
subplot(3,2,1); hold on;
plot(t_test, X_true(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, X_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Biomass (g/L)');
title('(A)');

% ---------- (B) Substrate ----------
subplot(3,2,2); hold on;
plot(t_test, S_true(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, S_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Substrate (g/L)');
title('(B)');

% ---------- (C) Total Protein ----------
subplot(3,2,3); hold on;
plot(t_test, Pt_true(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, Pt_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Total Protein (g/L)');
title('(C)');

% ---------- (D) Secreted Protein ----------
subplot(3,2,4); hold on;
plot(t_test, Pm_true(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, Pm_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Secreted Protein (g/L)');
title('(D)');

% ---------- (E) Volume ----------
subplot(3,2,5); hold on;
plot(t_test, V_meas(test_idx), 'k-', 'LineWidth', lw_true);
plot(t_test, V_pred_test,     'r--','LineWidth', lw_pred);
grid on; box on;
xlabel('Time (h)'); ylabel('Volume (L)');
title('(E)');

% ---------- (F) Feed rate ----------
subplot(3,2,6); hold on;
stairs(t_test, F_all(test_idx), 'm', 'LineWidth', 2.2);
grid on; box on;
xlabel('Time (h)'); ylabel('Feed rate (L/h)');
title('(F)');

% ---------- Legend (only once) ----------
subplot(3,2,1);
legend({'True','PINN'}, 'Location','best');

sgtitle(sprintf('Time-series comparison on Test batch %d (True vs PINN)', test_batch));

set(gcf,'Color','w');

%% 11. Fig-3 Style: Parity Plots + Physics Parity Plots (A–D)
% (A) Data Loss – Training:   Predicted vs Measured (scaled)
% (B) Data Loss – Validation: Predicted vs Measured (scaled)
% (C) Physics Loss – Collocation points:    Predicted dC/dt vs True dC/dt (scaled)
% (D) Physics Loss – Validation data pts:   Predicted dC/dt vs True dC/dt (scaled)

% ===================== Helper functions =====================
norm_fn_vis  = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);
denorm_fn    = @(y, s) double(extractdata(y)) .* (s(2) - s(1)) + s(1);
R2vis        = @(y,p)  1 - sum((y(:)-p(:)).^2) / (sum((y(:)-mean(y(:))).^2) + 1e-10);

% Central difference helper — same dt and logic as corrected physLoss
function [dX,dS,dPt,dPm] = centralDiff(net, inp, stats, dt)
    dt_norm  = dt / (stats.t(2) - stats.t(1));
    cP = inp; cP(1,:) = cP(1,:) + dt_norm;
    cM = inp; cM(1,:) = cM(1,:) - dt_norm;
    yP = extractdata(predict(net, dlarray(single(cP),'CB')));
    yM = extractdata(predict(net, dlarray(single(cM),'CB')));
    dX  = (yP(1,:) - yM(1,:)) * (stats.X(2)  - stats.X(1))  / (2*dt);
    dS  = (yP(2,:) - yM(2,:)) * (stats.S(2)  - stats.S(1))  / (2*dt);
    dPt = (yP(3,:) - yM(3,:)) * (stats.Pt(2) - stats.Pt(1)) / (2*dt);
    dPm = (yP(4,:) - yM(4,:)) * (stats.Pm(2) - stats.Pm(1)) / (2*dt);
end

dt_vis = 0.1;   % same corrected dt as training

% ===================== Scaled predictions =====================
pred_tr_raw  = extractdata(predict(net, dlInput));
pred_val_raw = extractdata(predict(net, dlInputVal));

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

% Scaled validation targets
X_val_sc  = pred_val_raw(1,:)';
S_val_sc  = pred_val_raw(2,:)';
Pt_val_sc = pred_val_raw(3,:)';
Pm_val_sc = pred_val_raw(4,:)';
V_val_sc  = norm_fn_vis(V_exact_val, stats.V);

X_val_m   = norm_fn_vis(X_meas(val_idx),  stats.X);
S_val_m   = norm_fn_vis(S_meas(val_idx),  stats.S);
Pt_val_m  = norm_fn_vis(Pt_meas(val_idx), stats.Pt);
Pm_val_m  = norm_fn_vis(Pm_meas(val_idx), stats.Pm);
V_val_m   = norm_fn_vis(V_exact_val, stats.V);

% R² (scaled)
r2X_tr=R2vis(X_tr_m,X_tr_sc); r2S_tr=R2vis(S_tr_m,S_tr_sc);
r2Pt_tr=R2vis(Pt_tr_m,Pt_tr_sc); r2Pm_tr=R2vis(Pm_tr_m,Pm_tr_sc);
r2X_v=R2vis(X_val_m,X_val_sc); r2S_v=R2vis(S_val_m,S_val_sc);
r2Pt_v=R2vis(Pt_val_m,Pt_val_sc); r2Pm_v=R2vis(Pm_val_m,Pm_val_sc);

% ===================== Physics at collocation (panel C) =====================
numColloc_vis = 600;
col = genColloc(numColloc_vis, t_train, X_train, S_train, ...
    Pt_train, Pm_train, V_train, F_train, stats);

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
[dX_pC,dS_pC,dPt_pC,dPm_pC] = centralDiff(net, col.input, stats, dt_vis);

% Scale by variable range for axis comparability
scX=stats.X(2)-stats.X(1); scS=stats.S(2)-stats.S(1);
scPt=stats.Pt(2)-stats.Pt(1); scPm=stats.Pm(2)-stats.Pm(1);

dX_tCsc=dX_tC/scX;  dS_tCsc=dS_tC/scS;
dPt_tCsc=dPt_tC/scPt; dPm_tCsc=dPm_tC/scPm;
dX_pCsc=dX_pC/scX;  dS_pCsc=dS_pC/scS;
dPt_pCsc=dPt_pC/scPt; dPm_pCsc=dPm_pC/scPm;

r2dX_c=R2vis(dX_tCsc,dX_pCsc); r2dS_c=R2vis(dS_tCsc,dS_pCsc);
r2dPt_c=R2vis(dPt_tCsc,dPt_pCsc); r2dPm_c=R2vis(dPm_tCsc,dPm_pCsc);

% ===================== Physics at validation pts (panel D) =====================
yV   = extractdata(predict(net, dlInputVal));
Xv   = yV(1,:).*(stats.X(2) -stats.X(1)) + stats.X(1);
Sv   = yV(2,:).*(stats.S(2) -stats.S(1)) + stats.S(1);
Ptv  = yV(3,:).*(stats.Pt(2)-stats.Pt(1))+ stats.Pt(1);
Pmv  = yV(4,:).*(stats.Pm(2)-stats.Pm(1))+ stats.Pm(1);
Vv   = max(V_exact_val', 0.1);
Dv   = F_all(val_idx)' ./ Vv;

Sv_s  = max(Sv, 1e-6);
mu_v  = 21.87*Sv_s./((Sv_s+0.4).*(Sv_s+62.5));
th_v  = 4.75*mu_v./(0.12+mu_v);
fp_v  = Sv_s.*exp(-5*Sv_s)./(0.1+Sv_s);

dX_tV  =  mu_v.*Xv   - Dv.*Xv;
dS_tV  = -Y*mu_v.*Xv + Dv.*(Sin-Sv);
dPt_tV =  fp_v.*Xv   - Dv.*Ptv;
dPm_tV =  th_v.*(Ptv-Pmv) - Dv.*Pmv;

[dX_pV,dS_pV,dPt_pV,dPm_pV] = centralDiff(net, inputVal, stats, dt_vis);

dX_tVsc=dX_tV/scX;  dS_tVsc=dS_tV/scS;
dPt_tVsc=dPt_tV/scPt; dPm_tVsc=dPm_tV/scPm;
dX_pVsc=dX_pV/scX;  dS_pVsc=dS_pV/scS;
dPt_pVsc=dPt_pV/scPt; dPm_pVsc=dPm_pV/scPm;

r2dX_v=R2vis(dX_tVsc,dX_pVsc); r2dS_v=R2vis(dS_tVsc,dS_pVsc);
r2dPt_v=R2vis(dPt_tVsc,dPt_pVsc); r2dPm_v=R2vis(dPm_tVsc,dPm_pVsc);

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

% ---- (B) Validation data parity ----
ax=subplot(2,2,2); hold(ax,'on');
scatter(ax,X_val_m,  X_val_sc,  ms,cX, 'o','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('X  (R²=%.3f)',r2X_v));
scatter(ax,S_val_m,  S_val_sc,  ms,cS, 's','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('S  (R²=%.3f)',r2S_v));
scatter(ax,Pt_val_m, Pt_val_sc, ms,cPt,'^','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('Pt (R²=%.3f)',r2Pt_v));
scatter(ax,Pm_val_m, Pm_val_sc, ms,cPm,'d','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('Pm (R²=%.3f)',r2Pm_v));
scatter(ax,V_val_m,  V_val_sc,  ms,cV, 'x','LineWidth',1.5,...
    'DisplayName','V (exact)');
plot(ax,[0 1],[0 1],'k-','LineWidth',2);
plot(ax,[0 1],[0.05 1.05],'k--','LineWidth',1); plot(ax,[0 1],[-0.05 0.95],'k--','LineWidth',1);
xlim(ax,[0 1]); ylim(ax,[-0.05 1.1]);
xlabel(ax,'Measured values (scaled)'); ylabel(ax,'Predicted values (scaled)');
title(ax,'(B) Data Loss: Validation','FontWeight','bold');
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

% ---- (D) Physics parity — validation data points ----
ax=subplot(2,2,4); hold(ax,'on');
scatter(ax,dX_tVsc(:), dX_pVsc(:), ms,cX, 'o','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('dX/dt  (R²=%.3f)',r2dX_v));
scatter(ax,dS_tVsc(:), dS_pVsc(:), ms,cS, 's','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('dS/dt  (R²=%.3f)',r2dS_v));
scatter(ax,dPt_tVsc(:),dPt_pVsc(:),ms,cPt,'^','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('dPt/dt (R²=%.3f)',r2dPt_v));
scatter(ax,dPm_tVsc(:),dPm_pVsc(:),ms,cPm,'d','filled','MarkerFaceAlpha',ma,...
    'DisplayName',sprintf('dPm/dt (R²=%.3f)',r2dPm_v));
allV=[dX_tVsc(:);dS_tVsc(:);dPt_tVsc(:);dPm_tVsc(:)];
lim_v=[min(allV)*1.1, max(allV)*1.1];
plot(ax,lim_v,lim_v,'k-','LineWidth',2);
xlim(ax,lim_v); ylim(ax,lim_v);
xlabel(ax,'True physics (scaled)'); ylabel(ax,'Predicted physics (scaled)');
title(ax,'(D) Physics Loss: Validation Data Points','FontWeight','bold');
legend(ax,'Location','northwest','FontSize',9); grid(ax,'on'); box(ax,'on');

sgtitle('Parity Plots: Data Loss (A,B) & Physics Loss (C,D)',...
    'FontSize',13,'FontWeight','bold');
set(gcf,'Color','w');


%% Functions

function [loss, grads, dL, pL] = lossFn(net, dlX, targets, colloc, lambda, Pm_phys_w, S_phys_w, stats, Sin, Y)
    dlY = forward(net, dlX); % Forward propagation — 4 outputs: X, S, Pt, Pm
    
    % Data loss: 4 variables only (V not predicted by network)
    w = [1, 1, 2, 5];
    dL = w(1)*mean((dlY(1,:) - targets.X' ).^2) + ...
         w(2)*mean((dlY(2,:) - targets.S' ).^2) + ...
         w(3)*mean((dlY(3,:) - targets.Pt').^2) + ...
         w(4)*mean((dlY(4,:) - targets.Pm').^2);
    dL = dL / sum(w);
    
    pL = physLoss(net, colloc, Pm_phys_w, S_phys_w, stats, Sin, Y); % Calculate physical loss
    
    loss = (1-lambda)*dL + lambda*pL; % Combined Total Loss
    grads = dlgradient(loss, net.Learnables); % Automatic differentiation for gradient computation
    
    dL = extractdata(dL); % Extract loss values for monitoring purposes
    pL = extractdata(pL);
end

function pL = physLoss(net, c, Pm_phys_w, S_phys_w, stats, Sin, Y)
    dlC = dlarray(single(c.input), 'CB');
    dlY = forward(net, dlC);  % 4 outputs: X, S, Pt, Pm
    
    % Min-Max denormalization: val = pred*(max-min) + min
    denorm = @(y, s) y .* (s(2) - s(1)) + s(1);
    
    X  = denorm(dlY(1,:), stats.X);
    S  = denorm(dlY(2,:), stats.S);
    Pt = denorm(dlY(3,:), stats.Pt);
    Pm = denorm(dlY(4,:), stats.Pm);
    
    % V_exact from collocation input row 6 (already denormalized in genColloc)
    V  = dlarray(single(c.V_raw), 'CB');
    
    Fr = c.F_raw;              % exact feed rate at collocation points
    D  = Fr ./ max(V, 0.1);   % exact dilution rate — no network error in D
    
    % Protect against S→0 numerical issues
    S_safe = max(S, 1e-6);
    % Park & Ramirez kinetics
    mu    = 21.87 * S_safe ./ ((S_safe + 0.4) .* (S_safe + 62.5));
    theta = 4.75  * mu     ./ (0.12 + mu);
    fp    = S_safe .* exp(-5 * S_safe) ./ (0.1 + S_safe);
    
    % Central finite-difference derivatives (perturb time input, row 1)
    % Use dt=0.1h to avoid float32 cancellation errors (dt_norm ≈ 0.0067 >> eps_float32)
    dt      = 0.1;
    dt_norm = dt / (stats.t(2) - stats.t(1));
    
    cPlus        = c.input;
    cPlus(1,:)   = cPlus(1,:) + dt_norm;
    cMinus       = c.input;
    cMinus(1,:)  = cMinus(1,:) - dt_norm;
    
    dlCp  = dlarray(single(cPlus),  'CB');
    dlCm  = dlarray(single(cMinus), 'CB');
    dlYp  = forward(net, dlCp);
    dlYm  = forward(net, dlCm);
    
    % Central difference: (f(t+h) - f(t-h)) / (2h)  — 2nd-order accurate
    dX  = (dlYp(1,:) - dlYm(1,:)) * (stats.X(2)  - stats.X(1))  / (2*dt);
    dS  = (dlYp(2,:) - dlYm(2,:)) * (stats.S(2)  - stats.S(1))  / (2*dt);
    dPt = (dlYp(3,:) - dlYm(3,:)) * (stats.Pt(2) - stats.Pt(1)) / (2*dt);
    dPm = (dlYp(4,:) - dlYm(4,:)) * (stats.Pm(2) - stats.Pm(1)) / (2*dt);
    % No dV — rV residual removed (V is exact, not learned)
    
    % Mass balance residuals (4 equations)
    rX  = dX  - (mu.*X  - D.*X);
    rS  = dS  - (-Y*mu.*X + D.*(Sin - S));
    rPt = dPt - (fp.*X  - D.*Pt);
    rPm = dPm - (theta.*(Pt - Pm) - D.*Pm);
    
    % Weighted normalised physics loss (4 terms)
    w_phys = [1, S_phys_w, 1.5, Pm_phys_w];
    
    pL = (w_phys(1)*(rX  / (stats.X(2)  - stats.X(1)) ).^2 + ...
          w_phys(2)*(rS  / (stats.S(2)  - stats.S(1)) ).^2 + ...
          w_phys(3)*(rPt / (stats.Pt(2) - stats.Pt(1))).^2 + ...
          w_phys(4)*(rPm / (stats.Pm(2) - stats.Pm(1))).^2);
    pL = mean(pL, 'all') / sum(w_phys);
end

function c = genColloc(n, t, X, S, Pt, Pm, V, F, stats)
    % Generate random collocation points within the physical range.
    % t contains multiple batches so timestamps are NOT unique — we cannot
    % call interp1 on the full t vector.  Instead, for each collocation
    % point we pick one random training point and perturb its time slightly
    % so that V and F remain physically consistent with that batch trajectory.

    % Pick n random indices from the training data
    idx    = randi(length(t), 1, n);
    t_raw  = t(idx)' + (rand(1, n) - 0.5) * 0.5;   % ±0.25 h jitter
    t_raw  = max(min(t_raw, max(t)), min(t));         % clamp to [t_min, t_max]

    % V and F taken from the same training point (physically consistent)
    V_raw  = max(V(idx)', 0.1);
    F_raw  = F(idx)';

    % State variables sampled independently across their full training range
    X_raw  = stats.X(1)  + (stats.X(2)  - stats.X(1))  * rand(1, n);
    S_raw  = stats.S(1)  + (stats.S(2)  - stats.S(1))  * rand(1, n);
    Pt_raw = stats.Pt(1) + (stats.Pt(2) - stats.Pt(1)) * rand(1, n);
    Pm_raw = stats.Pm(1) + (stats.Pm(2) - stats.Pm(1)) * rand(1, n);

    c.t_raw  = t_raw;  c.X_raw  = X_raw;  c.S_raw  = S_raw;
    c.Pt_raw = Pt_raw; c.Pm_raw = Pm_raw;
    c.V_raw  = V_raw;  c.F_raw  = F_raw;

    norm_fn = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);

    c.input = [
        norm_fn(t_raw,  stats.t);
        norm_fn(X_raw,  stats.X);
        norm_fn(S_raw,  stats.S);
        norm_fn(Pt_raw, stats.Pt);
        norm_fn(Pm_raw, stats.Pm);
        norm_fn(V_raw,  stats.V);   % V_exact from training point — row 6
        norm_fn(F_raw,  stats.F)
    ];

    c.F_norm = norm_fn(F_raw, stats.F);
end