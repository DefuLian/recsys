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

addpath(genpath('../recsys'))
if exist('maxk','file')
    rmpath(genpath('utils/MinMaxSelection'))
end