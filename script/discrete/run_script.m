parpool('local',5);
addpath(genpath('~/software/recsys'));
run_dmf('~/data/citeulikedata.mat', '~/result_qcf', true);
run_qcf('~/data/citeulikedata.mat', '~/result_qcf', true);
run_dmf('~/data/lastfmdata.mat', '~/result_qcf', true);
run_qcf('~/data/lastfmdata.mat', '~/result_qcf', true);

run_dmf('~/data/gowalladata.mat', '~/result_qcf', true);
run_qcf('~/data/gowalladata.mat', '~/result_qcf', true);
run_dmf('~/data/echonestdata.mat', '~/result_qcf', true);
run_qcf('~/data/echonestdata.mat', '~/result_qcf', true);


