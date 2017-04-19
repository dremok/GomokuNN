function [a isRandom] = qAgent(s,w,f,epsilon)
% assumes there is at least one legal move

% s -> b
n = sqrt(numel(s)-1);
b = reshape(s(1:end - 1),n,n);

isRandom = 0;
if all(b(:) == 0)
    % start in the middle:
    n = size(b,1);
    a = (n + 1)/2 * [1 1];
else
    if rand < epsilon
        isRandom = 1;
        
        % get possible actions (next to current stones):
        c = conv2(abs(b),ones(3),'same');
        m = (c > 0) & b == 0;
        [i j] = find(m);
        as = [i j];
    
        % choose random action:
        ii = randi(size(as,1),1,1);
        a = as(ii,:);
    else
        a = argmaxQ(s,f,w);
    end
end
