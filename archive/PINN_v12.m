clear; close all; clc;
%% 1. Data loading
data = readtable('data.csv');
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

%% 2. BATCH-LEVEL SPLIT (修改test batch选择)
fprintf('==========================================================\n');
fprintf('PINN v10 FIXED: Batch-Level Split with Diagnostics\n');
fprintf('==========================================================\n\n');

test_batch = 16;  
% Training set
train_batches_all = [16, 1, 4, 11, 15, 3, 12, 7, 8];
train_batches = setdiff(train_batches_all, test_batch);

% Validation set
val_batches = [5, 6, 2, 14, 10, 9, 13];

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

fprintf('\nVrange:\n');
fprintf('  Training set: [%.2f, %.2f]\n', V_train_range(1), V_train_range(2));
fprintf('  Test set: [%.2f, %.2f]', V_test_range(1), V_test_range(2));
fprintf('\n');

%% 5. Structure of Network
numInputs = 7;
numOutputs = 5;
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

%% 6. Training Strategy 
stage1_epochs = 2000;
lambda_stage1 = 0.1;
Pm_physics_weight_s1 = 0.5; % Enhance Pm weight
S_physics_weight_s1 = 1.0;  % Enhance S weight

stage2_epochs = 3000;
lambda_stage2_start = 0.1;
lambda_stage2_end = 0.5;
Pm_physics_weight_s2 = 1.5; % Enhance Pm weight
S_physics_weight_s2 = 1.5;  % Enhance S weight

stage3_epochs = 2000;
lambda_stage3 = 0.5;
Pm_physics_weight_s3 = 2.0; % Enhance Pm weight
S_physics_weight_s3 = 2.0;  % Enhance S weight

totalEpochs = stage1_epochs + stage2_epochs + stage3_epochs;
learningRate = 0.001;
numColloc = 350;

n_batches_per_epoch = round(length(train_batches) * 0.67);

fprintf('Training Strategy :\n');
fprintf('  Phase 1: λ=%.2f, Pm_w=%.1f, S_w=%.1f\n', ...
    lambda_stage1, Pm_physics_weight_s1, S_physics_weight_s1);
fprintf('  Phase 2: λ=%.2f→%.2f, Pm_w=%.1f, S_w=%.1f\n', ...
    lambda_stage2_start, lambda_stage2_end, Pm_physics_weight_s2, S_physics_weight_s2);
fprintf('  Phase 3: λ=%.2f, Pm_w=%.1f, S_w=%.1f\n', ...
    lambda_stage3, Pm_physics_weight_s3, S_physics_weight_s3);
fprintf('\n Stochastic Regularization: %d/%d batches per epoch\n\n', ...
    n_batches_per_epoch, length(train_batches));

%% 7. Prepare data (min-max)
norm_fn = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);

inputTrain = [
    norm_fn(t_train, stats.t), ...
    norm_fn(X_train, stats.X), ...
    norm_fn(S_train, stats.S), ...
    norm_fn(Pt_train, stats.Pt), ...
    norm_fn(Pm_train, stats.Pm), ...
    norm_fn(V_train, stats.V), ...    
    norm_fn(F_train, stats.F)
]';
dlInput = dlarray(single(inputTrain), 'CB');

targets.X = norm_fn(X_train, stats.X);
targets.S = norm_fn(S_train, stats.S);
targets.Pt = norm_fn(Pt_train, stats.Pt);
targets.Pm = norm_fn(Pm_train, stats.Pm);
targets.V = norm_fn(V_train, stats.V);

inputVal = [
    norm_fn(t_all(val_idx), stats.t), ...
    norm_fn(X_meas(val_idx), stats.X), ...
    norm_fn(S_meas(val_idx), stats.S), ...
    norm_fn(Pt_meas(val_idx), stats.Pt), ...
    norm_fn(Pm_meas(val_idx), stats.Pm), ...
    norm_fn(V_meas(val_idx), stats.V), ...   
    norm_fn(F_all(val_idx), stats.F)
]';
dlInputVal = dlarray(single(inputVal), 'CB');

inputTest = [
    norm_fn(t_all(test_idx), stats.t), ...
    norm_fn(X_meas(test_idx), stats.X), ...
    norm_fn(S_meas(test_idx), stats.S), ...
    norm_fn(Pt_meas(test_idx), stats.Pt), ...
    norm_fn(Pm_meas(test_idx), stats.Pm), ...
    norm_fn(V_meas(test_idx), stats.V), ...  
    norm_fn(F_all(test_idx), stats.F)
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
    if epoch <= stage1_epochs
        lambda = lambda_stage1;
        Pm_physics_weight = Pm_physics_weight_s1;
        S_physics_weight = S_physics_weight_s1;
        stage_name = 'S1';
    elseif epoch <= stage1_epochs + stage2_epochs
        progress = (epoch - stage1_epochs) / stage2_epochs;
        lambda = lambda_stage2_start + progress * (lambda_stage2_end - lambda_stage2_start);
        Pm_physics_weight = Pm_physics_weight_s2;
        S_physics_weight = S_physics_weight_s2;
        stage_name = 'S2';
    else
        lambda = lambda_stage3;
        Pm_physics_weight = Pm_physics_weight_s3;
        S_physics_weight = S_physics_weight_s3;
        stage_name = 'S3';
    end
    
    history.lambda(epoch) = lambda;
    
    selected_batch_ids = randperm(length(train_batches), n_batches_per_epoch);
    epoch_data_idx = [];
    for i = selected_batch_ids
        epoch_data_idx = [epoch_data_idx; batch_train_idx{i}];
    end
    
    dlInput_epoch = dlInput(:, epoch_data_idx);
    targets_epoch.X = targets.X(epoch_data_idx);
    targets_epoch.S = targets.S(epoch_data_idx);
    targets_epoch.Pt = targets.Pt(epoch_data_idx);
    targets_epoch.Pm = targets.Pm(epoch_data_idx);
    targets_epoch.V = targets.V(epoch_data_idx);
    
    colloc = genColloc(numColloc, t_train, X_train, S_train, ...
        Pt_train, Pm_train, V_train, F_train, stats);
    
    [loss, grads, dL, pL] = dlfeval(@lossFn, net, dlInput_epoch, ...
        targets_epoch, colloc, lambda, Pm_physics_weight, S_physics_weight, stats, Sin, Y);
    
    [net, avgGrad, avgSqGrad] = adamupdate(net, grads, avgGrad, ...
        avgSqGrad, epoch, learningRate);
    
    history.loss(epoch) = extractdata(loss);
    history.data(epoch) = dL;
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
X_pred = max(0, pred(1,:)' * (stats.X(2)-stats.X(1)) + stats.X(1));
S_pred = max(0, pred(2,:)' * (stats.S(2)-stats.S(1)) + stats.S(1));
Pt_pred = max(0, pred(3,:)' * (stats.Pt(2)-stats.Pt(1)) + stats.Pt(1));
Pm_pred = max(0, pred(4,:)' * (stats.Pm(2)-stats.Pm(1)) + stats.Pm(1));
V_pred = max(0, pred(5,:)' * (stats.V(2)-stats.V(1)) + stats.V(1));

fprintf('=== Training set (%d batches) ===\n', length(train_batches));
fprintf('  X:  R²=%.4f\n', R2(X_true(train_idx), X_pred));
fprintf('  S:  R²=%.4f\n', R2(S_true(train_idx), S_pred));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(train_idx), Pt_pred));
fprintf('  Pm: R²=%.4f \n', R2(Pm_true(train_idx), Pm_pred));
fprintf('  V:  R²=%.4f\n', R2(V_meas(train_idx), V_pred));

% Validation set
dlPredVal = predict(net, dlInputVal);
predVal = extractdata(dlPredVal);
X_pred_val = max(0, predVal(1,:)' * (stats.X(2)-stats.X(1)) + stats.X(1));
S_pred_val = max(0, predVal(2,:)' * (stats.S(2)-stats.S(1)) + stats.S(1));
Pt_pred_val = max(0, predVal(3,:)' * (stats.Pt(2)-stats.Pt(1)) + stats.Pt(1));
Pm_pred_val = max(0, predVal(4,:)' * (stats.Pm(2)-stats.Pm(1)) + stats.Pm(1));
V_pred_val = max(0, predVal(5,:)' * (stats.V(2)-stats.V(1)) + stats.V(1));

fprintf('\n=== Validation set (%d batches) ===\n', length(val_batches));
fprintf('  X:  R²=%.4f\n', R2(X_true(val_idx), X_pred_val));
fprintf('  S:  R²=%.4f\n', R2(S_true(val_idx), S_pred_val));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(val_idx), Pt_pred_val));
fprintf('  Pm: R²=%.4f \n', R2(Pm_true(val_idx), Pm_pred_val));
fprintf('  V:  R²=%.4f\n', R2(V_meas(val_idx), V_pred_val));

% Test set
dlPredTest = predict(net, dlInputTest);
predTest = extractdata(dlPredTest);
X_pred_test = max(0, predTest(1,:)' * (stats.X(2)-stats.X(1)) + stats.X(1));
S_pred_test = max(0, predTest(2,:)' * (stats.S(2)-stats.S(1)) + stats.S(1));
Pt_pred_test = max(0, predTest(3,:)' * (stats.Pt(2)-stats.Pt(1)) + stats.Pt(1));
Pm_pred_test = max(0, predTest(4,:)' * (stats.Pm(2)-stats.Pm(1)) + stats.Pm(1));
V_pred_test = max(0, predTest(5,:)' * (stats.V(2)-stats.V(1)) + stats.V(1));

fprintf('\n=== Test set (Batch %d) ===\n', test_batch);
fprintf('  X:  R²=%.4f\n', R2(X_true(test_idx), X_pred_test));
fprintf('  S:  R²=%.4f', R2(S_true(test_idx), S_pred_test));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(test_idx), Pt_pred_test));
fprintf('  Pm: R²=%.4f ', R2(Pm_true(test_idx), Pm_pred_test));
fprintf('  V:  R²=%.4f', R2(V_meas(test_idx), V_pred_test));

test_Pm_R2 = R2(Pm_true(test_idx), Pm_pred_test);
test_V_R2 = R2(V_meas(test_idx), V_pred_test);
fprintf('\n');
if test_Pm_R2 > 0.90 && test_V_R2 > 0.85
    fprintf('Test successful: PINN extrapolates well under new operating conditions.!\n\n');
elseif test_Pm_R2 > 0.80
    fprintf('️Testing Overall: Extrapolation capability is acceptable, but there is still room for improvement.\n\n');
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


%% Functions

function [loss, grads, dL, pL] = lossFn(net, dlX, targets, colloc, lambda, Pm_phys_w, S_phys_w, stats, Sin, Y)
    dlY = forward(net, dlX);
    
    w = [1, 1, 2, 5, 1];
    
    dL = w(1)*mean((dlY(1,:) - targets.X').^2) + ...
         w(2)*mean((dlY(2,:) - targets.S').^2) + ...
         w(3)*mean((dlY(3,:) - targets.Pt').^2) + ...
         w(4)*mean((dlY(4,:) - targets.Pm').^2) + ...
         w(5)*mean((dlY(5,:) - targets.V').^2);
    dL = dL / sum(w);
    
    pL = physLoss(net, colloc, Pm_phys_w, S_phys_w, stats, Sin, Y);
    
    loss = (1-lambda)*dL + lambda*pL;
    grads = dlgradient(loss, net.Learnables);
    
    dL = extractdata(dL);
    pL = extractdata(pL);
end

function pL = physLoss(net, c, Pm_phys_w, S_phys_w, stats, Sin, Y)
    dlC = dlarray(single(c.input), 'CB');
    dlY = forward(net, dlC);
    
    % Min-Max denormalization: val = pred * (max - min) + min
    denorm = @(y, s) y .* (s(2) - s(1)) + s(1);
    
    X = denorm(dlY(1,:), stats.X);
    S = denorm(dlY(2,:), stats.S);
    Pt = denorm(dlY(3,:), stats.Pt);
    Pm = denorm(dlY(4,:), stats.Pm);
    V = denorm(dlY(5,:), stats.V);
    
 
    Fr = denorm(c.F_norm, stats.F); 
    Fr = c.F_raw;
    D = Fr ./ max(V, 0.1);
    
    % Protection Mechanism: Preventing mu Calculation Errors Caused by Extremely Small S Values
    S_safe = max(S, 1e-6); 
    mu = 21.87 * S_safe ./ ((S_safe + 0.4) .* (S_safe + 62.5));
    theta = 4.75 * mu ./ (0.12 + mu);
    fp = S_safe .* exp(-5 * S_safe) ./ (0.1 + S_safe);
    
    dt = 0.01;
    % Compute the time derivative (Finite Difference)
    cPlus = c.input;

    dt_norm = dt / (stats.t(2) - stats.t(1));
    cPlus(1,:) = cPlus(1,:) + dt_norm;
    
    dlCp = dlarray(single(cPlus), 'CB');
    dlYp = forward(net, dlCp);
    
    dX = (dlYp(1,:)-dlY(1,:)) * (stats.X(2)-stats.X(1)) / dt;
    dS = (dlYp(2,:)-dlY(2,:)) * (stats.S(2)-stats.S(1)) / dt;
    dPt = (dlYp(3,:)-dlY(3,:)) * (stats.Pt(2)-stats.Pt(1)) / dt;
    dPm = (dlYp(4,:)-dlY(4,:)) * (stats.Pm(2)-stats.Pm(1)) / dt;
    dV = (dlYp(5,:)-dlY(5,:)) * (stats.V(2)-stats.V(1)) / dt;
    
    rX = dX - (mu.*X - D.*X);
    rS = dS - (-Y*mu.*X + D.*(Sin-S));
    rPt = dPt - (fp.*X - D.*Pt);
    rPm = dPm - (theta.*(Pt-Pm) - D.*Pm);
    rV = dV - Fr;
    
    w_phys = [1, S_phys_w, 1.5, Pm_phys_w, 1];
    
    pL = (w_phys(1)*(rX/(stats.X(2)-stats.X(1))).^2 + ...
          w_phys(2)*(rS/(stats.S(2)-stats.S(1))).^2 + ...
          w_phys(3)*(rPt/(stats.Pt(2)-stats.Pt(1))).^2 + ...
          w_phys(4)*(rPm/(stats.Pm(2)-stats.Pm(1))).^2 + ...
          w_phys(5)*(rV/(stats.V(2)-stats.V(1))).^2);
    pL = mean(pL, 'all') / sum(w_phys);
end

function c = genColloc(n, t, X, S, Pt, Pm, V, F, stats)
    % Generate random points within the physical range
    t_raw = min(t) + (max(t)-min(t))*rand(1,n);
    
    % Generate within the scope recorded in stats
    X_raw = stats.X(1) + (stats.X(2)-stats.X(1)) * rand(1,n);
    S_raw = stats.S(1) + (stats.S(2)-stats.S(1)) * rand(1,n);
    Pt_raw = stats.Pt(1) + (stats.Pt(2)-stats.Pt(1)) * rand(1,n);
    Pm_raw = stats.Pm(1) + (stats.Pm(2)-stats.Pm(1)) * rand(1,n);
    V_raw = stats.V(1) + (stats.V(2)-stats.V(1)) * rand(1,n);
    F_raw = stats.F(1) + (stats.F(2)-stats.F(1)) * rand(1,n);
    
    c.t_raw = t_raw; c.X_raw = X_raw; c.S_raw = S_raw;
    c.Pt_raw = Pt_raw; c.Pm_raw = Pm_raw; c.V_raw = V_raw; c.F_raw = F_raw;
    
    % Min-Max Normalized Input
    norm_fn = @(v, s) (v - s(1)) ./ (s(2) - s(1) + 1e-8);
    
    c.input = [
    norm_fn(t_raw, stats.t);
    norm_fn(X_raw, stats.X);
    norm_fn(S_raw, stats.S);
    norm_fn(Pt_raw, stats.Pt);
    norm_fn(Pm_raw, stats.Pm);
    norm_fn(V_raw, stats.V);    
    norm_fn(F_raw, stats.F)
    ];
    
    c.F_norm = norm_fn(F_raw, stats.F);
end