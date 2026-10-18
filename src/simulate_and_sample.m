function data = simulate_and_sample(theta, X0, V0, F_profile, t_meas, noise_rel)
% Integrate ODE and sample at t_meas.
% noise_rel: relative std (e.g. 0.05 for 5%). If 0, no noise.

x_init = [X0; V0];

odefun = @(t, x) logistic_fedbatch_ode(t, x, theta, F_profile);

% ODE solver settings (更稳一点)
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

% Integrate exactly on measurement grid (convenient)
[t, x] = ode45(odefun, t_meas, x_init, opts);

X_true = x(:,1);
V_true = x(:,2);

% 5% relative Gaussian noise on X
if noise_rel > 0
    sigma = noise_rel .* max(X_true, 1e-12); % 避免X接近0时sigma为0
    X_meas = X_true + sigma .* randn(size(X_true));
    % 防止出现负浓度（物理上不合理）
    X_meas = max(X_meas, 0);
else
    X_meas = X_true;
end

data.t = t;
data.X_true = X_true;
data.V_true = V_true;
data.X_meas = X_meas;
end
