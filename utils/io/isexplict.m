function v = isexplict(R)
v = max(R(R~=0)) > min(R(R~=0)) + 1e-3;
end