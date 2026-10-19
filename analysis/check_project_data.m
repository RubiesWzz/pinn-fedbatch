% Quick static check, without PINN training. Run from repository root.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
T = readtable(fullfile(projectRoot, 'data', 'data.csv'));
required = {'time_hr_', 'X_True_g_L_', 'X_noise_g_L_', 'S_True_g_L_', 'S_noise_g_L_', ...
    'PT_True_g_L_', 'PT_noise_g_L_', 'PM_True_g_L_', 'PM_noise_g_L_', ...
    'V_noise_L_', 'Feed_L_h_', 'batchNumber'};
assert(isempty(setdiff(required, T.Properties.VariableNames)), 'Required columns missing');
T = T(~isnan(T.time_hr_), :);
assert(all(ismember(1:16, unique(T.batchNumber)')), 'Expected batches 1:16');
fprintf('Data check passed: %d rows, %d batches.\n', height(T), numel(unique(T.batchNumber)));
