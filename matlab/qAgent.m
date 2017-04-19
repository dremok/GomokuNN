function [a isRandom phiAll] = qAgent(s,w,f,epsilon)
% assumes there is at least one legal move

% s -> b
n = sqrt(numel(s)-1);
b = reshape(s(1:end - 1),n,n);

isRandom = 0;
if all(b(:) == 0)
    % start in the middle:
    n = size(b,1);
    a = (n + 1)/2 * [1 1];
    phiAll = [1 a zeros(1,numel(w))];
else
    [a,~,phia,as]  = argmaxQ(s,f,w);
    if rand < epsilon
        isRandom = 1;
        
        % choose random action:
        ii = randi(size(as,1),1,1);
        a = as(ii,:);
    else
        
    end
    
    if size(as,1) > 100, as = as(1:100,:); end
    phiAll = [as'; phia];
    phiAll = [size(as,1) phiAll(:)'];
end
