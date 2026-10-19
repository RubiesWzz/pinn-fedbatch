function dx = logistic_fedbatch_ode(t, x, theta, F_profile)
% x(1)=X (g/L), x(2)=V (L)
X = x(1);
V = x(2);

F = feed_rate(t, F_profile);        % L/h
D = F / max(V, 1e-12);              % 1/h (avoid divide by zero)

mu = theta.mu_max * (1 - X/theta.X_max);

dX = mu * X - D * X;
dV = F;

dx = [dX; dV];
end