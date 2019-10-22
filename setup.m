% code snippet for compilation 
%!cd ./item_rec/alg
%mex -largeArrayDims piccf_sub.cpp
%mex -g -largeArrayDims piccf_sub.cpp
%!cd ./utils/MinMaxSelection
%mex -largeArrayDims inplacecolumnmex.c
%mex -largeArrayDims maxkmex.c
%mex -largeArrayDims minkmex.c
%mex -largeArrayDims releaseinplace.c
%!cd -

if exist('maxk')
    addpath(genpath('../recsys'))
    rmpath(genpath('utils/MinMaxSelection'))
else
    addpath(genpath('../recsys'))
end

if ~exist('annoy_mips')
    fprintf('compiling annoy_mips\n');
    mex -largeArrayDims utils/eval/annoy_mips.cpp -Iutils/eval/annoy/ -output utils/eval/annoy_mips
end

if ~exist('pq_search')
    fprintf('compiling pq_search\n');
    mex -largeArrayDims utils/eval/pq_search.cpp -output utils/eval/pq_search
end

if ~exist('topk_lookup')
    fprintf('compiling topk_lookup\n');
    mex -largeArrayDims utils/eval/topk_lookup.cpp -output utils/eval/topk_lookup
end

if ~exist('get_stat_fast')
    fprintf('compiling get_stat_fast\n');
    mex -largeArrayDims alg/discrete/qmf/get_stat_fast.cpp -output alg/discrete/qmf/get_stat_fast
end

if ~exist('ccd_bqp_mex')
    fprintf('compiling ccd_bqp_mex\n');
    mex -largeArrayDims alg/discrete/dmf/ccd_bqp_mex.c -output alg/discrete/dmf/ccd_bqp_mex
end

if ~exist('ccd_logit_mex')
    fprintf('compiling ccd_logit_mex\n');
    mex -largeArrayDims alg/discrete/dmf/ccd_logit_mex.cpp -output alg/discrete/dmf/ccd_logit_mex
end

if ~exist('iccf_sub')
    fprintf('compiling iccf_sub\n');
    mex -largeArrayDims alg/real/iccf/iccf_sub.cpp -output alg/real/iccf/iccf_sub
end

if ~exist('piccf_sub')
    fprintf('compiling piccf_sub\n');
    mex -largeArrayDims alg/real/piccf/piccf_sub.cpp -output alg/real/piccf/piccf_sub
end

if ~exist('apq_search')
    fprintf('compiling apq_search\n');
    mex -largeArrayDims utils/eval/apq_search.cpp -output utils/eval/apq_search
end
