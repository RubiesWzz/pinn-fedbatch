function r = residual_mu_only(mu, X0, V0, F_profile, t_meas, X_meas, Xmax_fixed)
theta.mu_max = mu;
theta.X_max  = Xmax_fixed;

sim = simulate_and_sample(theta, X0, V0, F_profile, t_meas, 0.0);
r = sim.X_true - X_meas;
end