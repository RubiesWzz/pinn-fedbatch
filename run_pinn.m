projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'));
run(fullfile(projectRoot, 'src', 'main_pinn.m'));
