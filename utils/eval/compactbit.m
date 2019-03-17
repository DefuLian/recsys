%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cb = compactbit(b)
%
% Written by Rob Fergus
% b = bits array
% cb = compacted string of bits (using words of 'word' bits)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
len = 32;
[nSamples, nbits] = size(b);
nwords = ceil(nbits/len);
cb = zeros([nSamples nwords], sprintf('uint%d',len));

for j = 1:nbits
    w = ceil(j/len);
    cb(:,w) = bitset(cb(:,w), mod(j-1,len)+1, b(:,j));
end