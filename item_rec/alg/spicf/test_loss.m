function loss = test_loss(Q, P)
QtQ = Q' * Q;
M = size(P,1);
loss = 0;
for i=1:M
    loss = loss +  trace( QtQ *P(i,:)'*P(i,:));
end
end