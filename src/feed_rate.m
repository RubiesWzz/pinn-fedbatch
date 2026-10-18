function F = feed_rate(t, F_profile)
% Piecewise-constant F(t) defined by breaks and values:
% breaks: [t0, t1, ..., tN], values: [F0, ..., F_{N-1}]
tb = F_profile.t_breaks(:)';
fv = F_profile.F_values(:)';

% Find segment index i such that tb(i) <= t < tb(i+1)
i = find(tb <= t, 1, 'last');
if isempty(i)
    i = 1;
end
if i >= numel(tb)
    i = numel(tb) - 1;
end

F = fv(i);
end