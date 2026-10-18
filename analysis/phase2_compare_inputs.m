%% phase2_compare_inputs.m
clear; clc; close all;
rng(1);

%% Nominal parameters and experiment setup
theta0.mu_max = 0.3;   % h^-1
theta0.X_max  = 47.3;  % g/L

X0 = 1.5;              % g/L
V0 = 2.5;              % L
T_end  = 24;           % h
t_meas = (0:1:T_end)'; % 1h sampling

% finite-difference relative step
rel_step = 1e-3;       % 0.1%

%% Define candidate feed profiles (all within [0.05, 0.5] L/h)
profiles = {};

% A) baseline
pA.name = 'A: baseline';
pA.t_breaks = [0, 8, 16, 24];
pA.F_values = [0.10, 0.30, 0.15];
profiles{end+1} = pA;

% B) lower dilution in late phase)
pB.name = 'B: low F late';
pB.t_breaks = [0, 8, 16, 24];
pB.F_values = [0.10, 0.30, 0.05];
profiles{end+1} = pB;

% C) high early, low late
pC.name = 'C: high early -> low late';
pC.t_breaks = [0, 6, 12, 24];
pC.F_values = [0.40, 0.20, 0.05];
profiles{end+1} = pC;


%% Loop: compute energies for each profile
nP = numel(profiles);
I_mu   = zeros(nP,1);
I_Xmax = zeros(nP,1);
Ratio  = zeros(nP,1);

% store normalized sensitivity curves
store_curves = true;
Sn_mu_all   = cell(nP,1);
Sn_Xmax_all = cell(nP,1);

for i = 1:nP
    F_profile = struct();
    F_profile.t_breaks = profiles{i}.t_breaks;
    F_profile.F_values = profiles{i}.F_values;

    out = compute_sensitivity_energy(theta0, X0, V0, F_profile, t_meas, rel_step);

    I_mu(i)   = out.I_mu;
    I_Xmax(i) = out.I_Xmax;
    Ratio(i)  = out.ratio;

    if store_curves
        Sn_mu_all{i}   = out.Sn_mu;
        Sn_Xmax_all{i} = out.Sn_Xmax;
    end

    fprintf('%s: I_mu=%.4e, I_Xmax=%.4e, ratio=%.4f\n', ...
        profiles{i}.name, I_mu(i), I_Xmax(i), Ratio(i));
end

%% Create comparison table
names = string(cellfun(@(p) p.name, profiles, 'UniformOutput', false))';
T = table(names, I_mu, I_Xmax, Ratio);
disp(T);

%% Bar plot: energies
figure;
bar([I_mu, I_Xmax]);
grid on;
set(gca, 'XTickLabel', names, 'XTickLabelRotation', 15);
ylabel('Normalized sensitivity energy  \Sigma_k \tilde S^2');
legend('I_{\mu_{max}}','I_{X_{max}}','Location','best');
title('Phase 2: sensitivity energy comparison across feed profiles');

%% Bar plot: ratio
figure;
bar(Ratio);
grid on;
set(gca, 'XTickLabel', names, 'XTickLabelRotation', 15);
ylabel('I_{\mu_{max}} / I_{X_{max}}');
title('Relative identifiability indicator (larger => mu_{max} dominates)');

%% plot normalized sensitivity curves
if store_curves
    figure; hold on; grid on;
    for i = 1:nP
        plot(t_meas, Sn_mu_all{i}, 'LineWidth', 1.2, 'DisplayName', names(i) + "  ~ S_{\mu}");
    end
    xlabel('t (h)'); ylabel('normalized sensitivity');
    title('\tilde S_{\mu_{max}}(t) across feed profiles');
    legend('Location','best');

    figure; hold on; grid on;
    for i = 1:nP
        plot(t_meas, Sn_Xmax_all{i}, 'LineWidth', 1.2, 'DisplayName', names(i) + "  ~ S_{Xmax}");
    end
    xlabel('t (h)'); ylabel('normalized sensitivity');
    title('S_{X_{max}}(t) across feed profiles');
    legend('Location','best');
end


%% Conclusion
% Three sets of experiments were conducted for comparison. 
% Input A: Segment constant, high F in the middle section, medium in the rear section
% Input B: Make the final segment F3 even smaller.
% Input C: In the initial phase, employ F to elevate D, thereby suppressing growth; subsequently, reduce F to release growth, enabling the trajectory to rapidly converge towards the plateau.

% This indicates that, under the considered experimental conditions, the biomass growth dynamics are mainly governed by the maximum specific growth rate.
% The carrying capacity Xmax only influences the system in the late phase and remains weakly identifiable.
% Consequently, classical parameter estimation is prone to over-parameterization.
% This motivates the use of physics-informed learning approaches, which can exploit the full trajectory information while alleviating identifiability issues.