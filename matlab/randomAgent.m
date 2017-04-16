function move = randomAgent(board)
% assumes there is at least one legal move


if all(board(:) == 0)
    % start in the middle:
    n = size(board,1);
    move = (n + 1)/2 * [1 1];
else
    % choose randomly next to current stones:
    c = conv2(board,ones(3),'same');
    m = (c > 0) & board == 0;
    [i j] = find(m);
    ii = randi(numel(i),1,1);
    move = [i(ii) j(ii)];
end
