function out = compute_sensitivity_energy(theta0, X0, V0, F_profile, t_meas, rel_step)
% Compute normalized sensitivities and their energy for theta=[mu_max, X_max]
% using central finite differences.

% nominal simulation (no noise)
sim0 = simulate_and_sample(theta0, X0, V0, F_profile, t_meas, 0.0);
X_nom = sim0.X_true;

% FD steps (relative)
dmu   = rel_step * max(theta0.mu_max, 1e-6);
dXmax = rel_step * max(theta0.X_max,  1e-6);

% S_mu
theta_p = theta0; theta_m = theta0;
theta_p.mu_max = theta0.mu_max + dmu;
theta_m.mu_max = theta0.mu_max - dmu;
Xp = simulate_and_sample(theta_p, X0, V0, F_profile, t_meas, 0.0).X_true;
Xm = simulate_and_sample(theta_m, X0, V0, F_profile, t_meas, 0.0).X_true;
S_mu = (Xp - Xm) / (2*dmu);

% S_Xmax
theta_p = theta0; theta_m = theta0;
theta_p.X_max = theta0.X_max + dXmax;
theta_m.X_max = theta0.X_max - dXmax;
Xp = simulate_and_sample(theta_p, X0, V0, F_profile, t_meas, 0.0).X_true;
Xm = simulate_and_sample(theta_m, X0, V0, F_profile, t_meas, 0.0).X_true;
S_Xmax = (Xp - Xm) / (2*dXmax);

% normalized sensitivities
epsX = 1e-9;
Sn_mu   = (theta0.mu_max ./ max(X_nom, epsX)) .* S_mu;
Sn_Xmax = (theta0.X_max  ./ max(X_nom, epsX)) .* S_Xmax;

% energies
I_mu   = sum(Sn_mu.^2);
I_Xmax = sum(Sn_Xmax.^2);
ratio  = I_mu / max(I_Xmax, 1e-12);

% pack outputs
out = struct();
out.t = t_meas;
out.X_nom = X_nom;

out.S_mu = S_mu;
out.S_Xmax = S_Xmax;

out.Sn_mu = Sn_mu;
out.Sn_Xmax = Sn_Xmax;

out.I_mu = I_mu;
out.I_Xmax = I_Xmax;
out.ratio = ratio;
end
