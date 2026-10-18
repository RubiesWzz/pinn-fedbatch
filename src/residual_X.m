function r = residual_X(p, X0, V0, F_profile, t_meas, X_meas)
% residual for lsqnonlin: r_k = X_sim(t_k; p) - X_meas(t_k)

theta.mu_max = p(1);
theta.X_max  = p(2);

sim = simulate_and_sample(theta, X0, V0, F_profile, t_meas, 0.0);

r = sim.X_true - X_meas;
end
