function [a Qa] = argmaxQ(s,f,w)

% s -> b
n = sqrt(numel(s)-1);
b = reshape(s(1:end - 1),n,n);

% get possible actions (next to current stones):
c = conv2(abs(b),ones(3),'same');
m = (c > 0) & b == 0;
[i j] = find(m);
as = [i j];

% get Q(s,a) for all possible a
nActions = size(as,1);
Q = zeros(nActions,1);
for j = 1:nActions
    Q(j) = w'*phi(s,as(j,:),f);
end

% find the best action (choose randomly among equally good ones):
k = find(Q == max(Q));
if numel(k) > 1, k = k(randi(numel(k),1,1)); end
a = as(k,:);
Qa = Q(k,:);
