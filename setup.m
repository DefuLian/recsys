!cd ./item_rec/alg
mex -largeArrayDims piccf_sub.cpp
mex -g -largeArrayDims piccf_sub.cpp
!cd ./utils/MinMaxSelection
mex -largeArrayDims inplacecolumnmex.c
mex -largeArrayDims maxkmex.c
mex -largeArrayDims minkmex.c
mex -largeArrayDims releaseinplace.c
!cd -