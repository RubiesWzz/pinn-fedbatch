%% PINN v9 - 修复Physics Loss根本性错误
% =========================================================================
% v8问题：Physics loss使用随机配点值，而非网络预测值
% v9改进：Physics loss全部基于网络预测值
% 
% 正确的physics residual：
% rPm = dPm/dt - (θ(Pt-Pm) - D*Pm)
% 其中所有量（dPm, Pt, Pm, θ）都来自网络预测！
% =========================================================================

clear; close all; clc;

%% 1. 数据加载
fprintf('=== PINN v9 - 修复Physics Loss ===\n');
fprintf('关键修复:\n');
fprintf('  - Physics loss全部基于网络预测\n');
fprintf('  - 不再使用随机配点值Ptr, Pmr\n');
fprintf('  - 真正的物理约束：dPm = θ(Pt-Pm) - D*Pm\n\n');

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

%% 2. 数据分割
rng(42);

train_idx = false(size(t_all));
val_idx = false(size(t_all));

for b = 1:max(batch_num)
    b_idx = find(batch_num == b);
    n_b = length(b_idx);
    perm = randperm(n_b);
    n_train = round(0.7 * n_b);
    train_idx(b_idx(perm(1:n_train))) = true;
    val_idx(b_idx(perm(n_train+1:end))) = true;
end

t_train = t_all(train_idx);
X_train = X_meas(train_idx);
S_train = S_meas(train_idx);
Pt_train = Pt_meas(train_idx);
Pm_train = Pm_meas(train_idx);
V_train = V_meas(train_idx);
F_train = F_all(train_idx);

fprintf('Train: %d, Val: %d\n\n', sum(train_idx), sum(val_idx));

%% 3. Z-score归一化
stats.t = [mean(t_train), std(t_train)];
stats.X = [mean(X_train), std(X_train)];
stats.S = [mean(S_train), std(S_train)];
stats.V = [mean(V_train), std(V_train)];
stats.F = [mean(F_train), max(std(F_train), 0.1)];
stats.Pt = [mean(Pt_train), std(Pt_train) + 0.01];
stats.Pm = [mean(Pm_train), std(Pm_train) + 0.01];

fprintf('Z-score统计:\n');
fprintf('  Pm: mean=%.3f, std=%.3f\n\n', stats.Pm(1), stats.Pm(2));

%% 4. 网络架构
numInputs = 6;
numOutputs = 5;
hiddenSize = 80;

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
fprintf('网络参数: %d\n\n', sum(cellfun(@numel, net.Learnables.Value)));

%% 5. 训练策略
stage1_epochs = 1500;
%stage1_epochs = 1000;
lambda_stage1 = 0.1;
Pm_physics_weight_s1 = 0.5;

stage2_epochs = 2000;
%stage2_epochs = 1000;
lambda_stage2_start = 0.1;
lambda_stage2_end = 0.4;
Pm_physics_weight_s2 = 1.5;

stage3_epochs = 1500;
%stage3_epochs = 1000;
lambda_stage3 = 0.4;
Pm_physics_weight_s3 = 2.0;

totalEpochs = stage1_epochs + stage2_epochs + stage3_epochs;
learningRate = 0.001;
numColloc = 350;

fprintf('训练策略:\n');
fprintf('  阶段1: λ=%.2f, Pm_phys_w=%.1f\n', lambda_stage1, Pm_physics_weight_s1);
fprintf('  阶段2: λ=%.2f→%.2f, Pm_phys_w=%.1f\n', lambda_stage2_start, lambda_stage2_end, Pm_physics_weight_s2);
fprintf('  阶段3: λ=%.2f, Pm_phys_w=%.1f\n\n', lambda_stage3, Pm_physics_weight_s3);

%% 6. 准备数据
inputTrain = [
    (t_train - stats.t(1)) / stats.t(2), ...
    (X_train - stats.X(1)) / stats.X(2), ...
    (S_train - stats.S(1)) / stats.S(2), ...
    (Pt_train - stats.Pt(1)) / stats.Pt(2), ...
    (Pm_train - stats.Pm(1)) / stats.Pm(2), ...
    (F_train - stats.F(1)) / stats.F(2)
]';

dlInput = dlarray(single(inputTrain), 'CB');

targets.X = (X_train - stats.X(1)) / stats.X(2);
targets.S = (S_train - stats.S(1)) / stats.S(2);
targets.Pt = (Pt_train - stats.Pt(1)) / stats.Pt(2);
targets.Pm = (Pm_train - stats.Pm(1)) / stats.Pm(2);
targets.V = (V_train - stats.V(1)) / stats.V(2);

inputVal = [
    (t_all(val_idx) - stats.t(1)) / stats.t(2), ...
    (X_meas(val_idx) - stats.X(1)) / stats.X(2), ...
    (S_meas(val_idx) - stats.S(1)) / stats.S(2), ...
    (Pt_meas(val_idx) - stats.Pt(1)) / stats.Pt(2), ...
    (Pm_meas(val_idx) - stats.Pm(1)) / stats.Pm(2), ...
    (F_all(val_idx) - stats.F(1)) / stats.F(2)
]';
dlInputVal = dlarray(single(inputVal), 'CB');

%% 7. 训练循环
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
max_patience = 1500;

fprintf('训练中...\n');
tic;

for epoch = 1:totalEpochs
    if epoch <= stage1_epochs
        lambda = lambda_stage1;
        Pm_physics_weight = Pm_physics_weight_s1;
        stage_name = 'S1';
    elseif epoch <= stage1_epochs + stage2_epochs
        progress = (epoch - stage1_epochs) / stage2_epochs;
        lambda = lambda_stage2_start + progress * (lambda_stage2_end - lambda_stage2_start);
        Pm_physics_weight = Pm_physics_weight_s2;
        stage_name = 'S2';
    else
        lambda = lambda_stage3;
        Pm_physics_weight = Pm_physics_weight_s3;
        stage_name = 'S3';
    end
    
    history.lambda(epoch) = lambda;
    
    % 生成配点
    colloc = genColloc(numColloc, t_train, X_train, S_train, ...
        Pt_train, Pm_train, V_train, F_train, stats);
    
    % 训练步（修复的physics loss）⭐
    [loss, grads, dL, pL] = dlfeval(@lossFn, net, dlInput, ...
        targets, colloc, lambda, Pm_physics_weight, stats, Sin, Y);
    
    [net, avgGrad, avgSqGrad] = adamupdate(net, grads, avgGrad, ...
        avgSqGrad, epoch, learningRate);
    
    history.loss(epoch) = extractdata(loss);
    history.data(epoch) = dL;
    history.physics(epoch) = pL;
    
    % 每100 epochs评估
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
                fprintf(' ⭐\n');
            else
                fprintf('\n');
            end
        end
    end
    
    if patience >= max_patience
        fprintf('\n提前停止于Epoch %d (最佳Pm验证R²=%.4f)\n', epoch, best_val_Pm);
        break;
    end
end
fprintf('完成: %.1f秒\n', toc);
fprintf('最佳Pm验证R²: %.4f\n\n', best_val_Pm);

net = best_net;

%% 8. 最终评估
dlPred = predict(net, dlInput);
pred = extractdata(dlPred);

X_pred = pred(1,:)' * stats.X(2) + stats.X(1);
S_pred = pred(2,:)' * stats.S(2) + stats.S(1);
Pt_pred = pred(3,:)' * stats.Pt(2) + stats.Pt(1);
Pm_pred = pred(4,:)' * stats.Pm(2) + stats.Pm(1);
V_pred = pred(5,:)' * stats.V(2) + stats.V(1);

X_pred = max(0, X_pred);
S_pred = max(0, S_pred);
Pt_pred = max(0, Pt_pred);
Pm_pred = max(0, Pm_pred);
V_pred = max(0, V_pred);

R2 = @(y,p) 1 - sum((y-p).^2)/(sum((y-mean(y)).^2)+1e-10);

fprintf('=== 训练集 ===\n');
fprintf('  X:  R²=%.4f\n', R2(X_true(train_idx), X_pred));
fprintf('  S:  R²=%.4f\n', R2(S_true(train_idx), S_pred));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(train_idx), Pt_pred));
fprintf('  Pm: R²=%.4f ⭐\n', R2(Pm_true(train_idx), Pm_pred));

dlPredVal = predict(net, dlInputVal);
predVal = extractdata(dlPredVal);

X_pred_val = predVal(1,:)' * stats.X(2) + stats.X(1);
S_pred_val = predVal(2,:)' * stats.S(2) + stats.S(1);
Pt_pred_val = predVal(3,:)' * stats.Pt(2) + stats.Pt(1);
Pm_pred_val = predVal(4,:)' * stats.Pm(2) + stats.Pm(1);

X_pred_val = max(0, X_pred_val);
S_pred_val = max(0, S_pred_val);
Pt_pred_val = max(0, Pt_pred_val);
Pm_pred_val = max(0, Pm_pred_val);

fprintf('\n=== 验证集 ===\n');
fprintf('  X:  R²=%.4f\n', R2(X_true(val_idx), X_pred_val));
fprintf('  S:  R²=%.4f\n', R2(S_true(val_idx), S_pred_val));
fprintf('  Pt: R²=%.4f\n', R2(Pt_true(val_idx), Pt_pred_val));
fprintf('  Pm: R²=%.4f ⭐\n', R2(Pm_true(val_idx), Pm_pred_val));

%% 9. 可视化

% 图1: Loss曲线
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

sgtitle(sprintf('Loss Evolution (Y=%.1f, λ=%.2f)', Y, lambda));
saveas(gcf, 'loss_v3.png');

% 图2: Parity plots - 训练和验证合并显示
figure('Position', [100 100 1200 800]);

subplot(2,2,1);
scatter(X_true(train_idx), X_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(X_true(val_idx), X_pred_val, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
plot([0 3], [0 3], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('Biomass X\nTrain R²=%.3f, Val R²=%.3f', ...
    R2(X_true(train_idx), X_pred), R2(X_true(val_idx), X_pred_val)));
legend('Train', 'Val', 'Location', 'best'); grid on;

subplot(2,2,2);
scatter(S_true(train_idx), S_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(S_true(val_idx), S_pred_val, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
plot([0 20], [0 20], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('Substrate S\nTrain R²=%.3f, Val R²=%.3f', ...
    R2(S_true(train_idx), S_pred), R2(S_true(val_idx), S_pred_val)));
legend('Train', 'Val', 'Location', 'best'); grid on;

subplot(2,2,3);
scatter(Pt_true(train_idx), Pt_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(Pt_true(val_idx), Pt_pred_val, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
plot([0 5], [0 5], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('Total Protein Pt\nTrain R²=%.3f, Val R²=%.3f', ...
    R2(Pt_true(train_idx), Pt_pred), R2(Pt_true(val_idx), Pt_pred_val)));
legend('Train', 'Val', 'Location', 'best'); grid on;

subplot(2,2,4);
scatter(Pm_true(train_idx), Pm_pred, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(Pm_true(val_idx), Pm_pred_val, 40, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
plot([0 4], [0 4], 'k--', 'LineWidth', 2);
xlabel('True (g/L)'); ylabel('Predicted (g/L)');
title(sprintf('Secreted Protein Pm\nTrain R²=%.3f, Val R²=%.3f', ...
    R2(Pm_true(train_idx), Pm_pred), R2(Pm_true(val_idx), Pm_pred_val)));
legend('Train', 'Val', 'Location', 'best'); grid on;

sgtitle('Predictions vs True Values');
saveas(gcf, 'parity_v3.png');

% 图3: 多个batch的时间序列
% 图3: 多个batch的时间序列
figure('Position', [100 100 1600 800]);

% ✨ 修改这里：指定要显示的batch
batches_show = [1, 4, 7, 10, 13, 16];
% 确保这些batch存在于数据中
batches_show = batches_show(ismember(batches_show, unique(batch_num)));

for i = 1:length(batches_show)
    b = batches_show(i);
    % 找到这个batch的所有点
    b_all_idx = batch_num == b;
    t_b = t_all(b_all_idx);
    [t_sorted, si] = sort(t_b);
    
    % 真实值
    X_b_true = X_true(b_all_idx); X_b_true = X_b_true(si);
    S_b_true = S_true(b_all_idx); S_b_true = S_b_true(si);
    Pt_b_true = Pt_true(b_all_idx); Pt_b_true = Pt_b_true(si);
    Pm_b_true = Pm_true(b_all_idx); Pm_b_true = Pm_b_true(si);
    
    % 预测 - 需要对整个batch预测
    X_b_meas = X_meas(b_all_idx); X_b_meas = X_b_meas(si);
    S_b_meas = S_meas(b_all_idx); S_b_meas = S_b_meas(si);
    Pt_b_meas = Pt_meas(b_all_idx); Pt_b_meas = Pt_b_meas(si);
    Pm_b_meas = Pm_meas(b_all_idx); Pm_b_meas = Pm_b_meas(si);
    F_b = F_all(b_all_idx); F_b = F_b(si);
    
    inputB = [
        (t_sorted - stats.t(1)) / stats.t(2), ...
        (X_b_meas - stats.X(1)) / stats.X(2), ...
        (S_b_meas - stats.S(1)) / stats.S(2), ...
        (Pt_b_meas - stats.Pt(1)) / stats.Pt(2), ...
        (Pm_b_meas - stats.Pm(1)) / stats.Pm(2), ...
        (F_b - stats.F(1)) / stats.F(2)
    ]';
    dlInputB = dlarray(single(inputB), 'CB');
    dlPredB = predict(net, dlInputB);
    predB = extractdata(dlPredB);
    
    X_b_pred = predB(1,:)' * stats.X(2) + stats.X(1);
    S_b_pred = predB(2,:)' * stats.S(2) + stats.S(1);
    Pt_b_pred = predB(3,:)' * stats.Pt(2) + stats.Pt(1);
    Pm_b_pred = predB(4,:)' * stats.Pm(2) + stats.Pm(1);
    
    % 绘图
    subplot(2,3,i);
    yyaxis left;
    plot(t_sorted, X_b_true, 'b-', 'LineWidth', 2);
    hold on;
    plot(t_sorted, X_b_pred, 'b--', 'LineWidth', 1.5);
    ylabel('X, Pt, Pm (g/L)');
    plot(t_sorted, Pt_b_true, 'g-', 'LineWidth', 2);
    plot(t_sorted, Pt_b_pred, 'g--', 'LineWidth', 1.5);
    plot(t_sorted, Pm_b_true, 'm-', 'LineWidth', 2);
    plot(t_sorted, Pm_b_pred, 'm--', 'LineWidth', 1.5);
    
    yyaxis right;
    plot(t_sorted, S_b_true, 'r-', 'LineWidth', 2);
    plot(t_sorted, S_b_pred, 'r--', 'LineWidth', 1.5);
    ylabel('S (g/L)');
    
    xlabel('Time (h)');
    title(sprintf('Batch %d', b));
    grid on;
    
    if i == 1
        legend('X true', 'X pred', 'Pt true', 'Pt pred', ...
               'Pm true', 'Pm pred', 'S true', 'S pred', ...
               'Location', 'best', 'FontSize', 7);
    end
end
sgtitle('Time Profiles for Different Batches (solid=true, dashed=PINN)');
saveas(gcf, 'batches_v3.png');

% 图4: 残差分析
figure('Position', [100 100 1200 600]);

subplot(2,4,1);
histogram(X_true(train_idx) - X_pred, 20);
xlabel('Residual'); title('X Train Residuals');

subplot(2,4,2);
histogram(S_true(train_idx) - S_pred, 20);
xlabel('Residual'); title('S Train Residuals');

subplot(2,4,3);
histogram(Pt_true(train_idx) - Pt_pred, 20);
xlabel('Residual'); title('Pt Train Residuals');

subplot(2,4,4);
histogram(Pm_true(train_idx) - Pm_pred, 20);
xlabel('Residual'); title('Pm Train Residuals');

subplot(2,4,5);
histogram(X_true(val_idx) - X_pred_val, 20);
xlabel('Residual'); title('X Val Residuals');

subplot(2,4,6);
histogram(S_true(val_idx) - S_pred_val, 20);
xlabel('Residual'); title('S Val Residuals');

subplot(2,4,7);
histogram(Pt_true(val_idx) - Pt_pred_val, 20);
xlabel('Residual'); title('Pt Val Residuals');

subplot(2,4,8);
histogram(Pm_true(val_idx) - Pm_pred_val, 20);
xlabel('Residual'); title('Pm Val Residuals');

sgtitle('Residual Distributions');
saveas(gcf, 'residuals_v3.png');

fprintf('\n图表已保存!\n');


%% 函数

function [loss, grads, dL, pL] = lossFn(net, dlX, targets, colloc, lambda, Pm_phys_w, stats, Sin, Y)
    dlY = forward(net, dlX);
    
    w = [1, 1, 2, 5, 1];
    
    dL = w(1)*mean((dlY(1,:) - targets.X').^2) + ...
         w(2)*mean((dlY(2,:) - targets.S').^2) + ...
         w(3)*mean((dlY(3,:) - targets.Pt').^2) + ...
         w(4)*mean((dlY(4,:) - targets.Pm').^2) + ...
         w(5)*mean((dlY(5,:) - targets.V').^2);
    dL = dL / sum(w);
    
    % 修复的physics loss ⭐⭐⭐
    pL = physLoss(net, colloc, Pm_phys_w, stats, Sin, Y);
    
    loss = (1-lambda)*dL + lambda*pL;
    grads = dlgradient(loss, net.Learnables);
    
    dL = extractdata(dL);
    pL = extractdata(pL);
end

function pL = physLoss(net, c, Pm_phys_w, stats, Sin, Y)
    % 在配点处评估网络 ⭐
    dlC = dlarray(single(c.input), 'CB');
    dlY = forward(net, dlC);
    
    % 网络预测的值（在配点处）⭐
    X = dlY(1,:)*stats.X(2) + stats.X(1);
    S = dlY(2,:)*stats.S(2) + stats.S(1);
    Pt = dlY(3,:)*stats.Pt(2) + stats.Pt(1);
    Pm = dlY(4,:)*stats.Pm(2) + stats.Pm(1);
    V = dlY(5,:)*stats.V(2) + stats.V(1);
    
    % 外部输入（F）从配点获取（因为不是状态变量）
    Fr = c.F_raw;
    Vr = c.V_raw;  % 也用原始V计算D（因为V的物理方程是独立的）
    
    % 物理参数（基于网络预测的S）⭐
    D = Fr ./ max(Vr, 0.1);
    mu = 21.87*S./((S+0.4).*(S+62.5));    % ← 用网络预测的S
    theta = 4.75*mu./(0.12+mu);
    fp = S.*exp(-5*S)./(0.1+S);          % ← 用网络预测的S
    
    % 时间导数（基于网络预测）⭐
    dt = 0.01;
    cPlus = c.input;
    cPlus(1,:) = (c.t_raw + dt - stats.t(1))/stats.t(2);
    dlCp = dlarray(single(cPlus), 'CB');
    dlYp = forward(net, dlCp);
    
    dX = (dlYp(1,:)-dlY(1,:))*stats.X(2)/dt;
    dS = (dlYp(2,:)-dlY(2,:))*stats.S(2)/dt;
    dPt = (dlYp(3,:)-dlY(3,:))*stats.Pt(2)/dt;
    dPm = (dlYp(4,:)-dlY(4,:))*stats.Pm(2)/dt;
    dV = (dlYp(5,:)-dlY(5,:))*stats.V(2)/dt;
    
    % 残差（全部基于网络预测）⭐⭐⭐
    rX = dX - (mu.*X - D.*X);                    % ← 用网络预测的X
    rS = dS - (-Y*mu.*X + D.*(Sin-S));          % ← 用网络预测的X和S
    rPt = dPt - (fp.*X - D.*Pt);                 % ← 用网络预测的X和Pt
    rPm = dPm - (theta.*(Pt-Pm) - D.*Pm);       % ← 用网络预测的Pt和Pm ⭐
    rV = dV - Fr;
    
    w_phys = [1, 1, 1.5, Pm_phys_w, 1];
    
    pL = (w_phys(1)*(rX/stats.X(2)).^2 + ...
          w_phys(2)*(rS/stats.S(2)).^2 + ...
          w_phys(3)*(rPt/stats.Pt(2)).^2 + ...
          w_phys(4)*(rPm/stats.Pm(2)).^2 + ...
          w_phys(5)*(rV/stats.V(2)).^2);
    pL = mean(pL, 'all') / sum(w_phys);
end

function c = genColloc(n, t, X, S, Pt, Pm, V, F, stats)
    c.t_raw = min(t) + (max(t)-min(t))*rand(1,n);
    c.X_raw = max(0.1,min(X)*0.8) + (max(X)*1.2-max(0.1,min(X)*0.8))*rand(1,n);
    c.S_raw = max(0.01,min(S)*0.5) + (max(S)*1.1-max(0.01,min(S)*0.5))*rand(1,n);
    c.Pt_raw = max(Pt)*1.2*rand(1,n);
    c.Pm_raw = max(Pm)*1.2*rand(1,n);
    c.V_raw = max(0.5,min(V)*0.9) + (max(V)*1.1-max(0.5,min(V)*0.9))*rand(1,n);
    c.F_raw = min(F) + (max(F)-min(F))*rand(1,n);
    
    c.input = [
        (c.t_raw-stats.t(1))/stats.t(2);
        (c.X_raw-stats.X(1))/stats.X(2);
        (c.S_raw-stats.S(1))/stats.S(2);
        (c.Pt_raw-stats.Pt(1))/stats.Pt(2);
        (c.Pm_raw-stats.Pm(1))/stats.Pm(2);
        (c.F_raw-stats.F(1))/stats.F(2)
    ];
end