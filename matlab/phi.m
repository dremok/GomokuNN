function phi_ = phi(s,a,f)
% s: state, [board(:); pToPlay; pMyToken]
% a: action
% f: features

% phi(s,a,f) := feature counts, after the move a

n = sqrt(numel(s)-1);
b = reshape(s(1:end - 1),n,n);
p = s(end);
assert(b(a(1),a(2)) == 0,'illegal move')
b(a(1),a(2)) = p;
phi_ = countpattern(b,p,f);