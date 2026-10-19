% Run the physics-only ODE baseline from the repository root.
projectRoot = fileparts(mfilename('fullpath'));
cd(projectRoot);
addpath(fullfile(projectRoot, 'src'));
Pureode_baseline;
