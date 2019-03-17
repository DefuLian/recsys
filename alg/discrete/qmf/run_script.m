parpool('local',5);
addpath(genpath('~/software/recsys'));
run_qcf('~/data/amazondata.mat', '~/result_qcf', true);
run_qcf('~/data/yelpdata.mat', '~/result_qcf', true);
run_qcf('~/data/ml10Mdata.mat', '~/result_qcf', true);
run_qcf('~/data/netflixdata.mat', '~/result_qcf', true);