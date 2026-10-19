% Run PINN V15 from the repository root (MATLAB Deep Learning Toolbox required).
% The script sets the current directory to the folder containing this file.
projectRoot = fileparts(mfilename('fullpath'));
cd(projectRoot);
addpath(fullfile(projectRoot, 'src'));
PINN_V15;
