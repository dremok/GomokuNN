function move = qAgent(board,player,w,f,epsilon)
% assumes there is at least one legal move

if all(board(:) == 0)
    % start in the middle:
    n = size(board,1);
    move = (n + 1)/2 * [1 1];
else
    % get possible actions (next to current stones):
    c = conv2(abs(board),ones(3),'same');
    m = (c > 0) & board == 0;
    [i j] = find(m);
    moves = [i j];
    
    if rand < epsilon
        % take random action:
        ii = randi(size(moves,1),1,1);
        move = [i(ii) j(ii)];
    else
        % evaluate Q(s,a) for all actions:
        Qa = NaN(size(moves,1),1);
        for j = 1:size(moves,1)
            Qa(j) = w'*phi(board,moves(j,:),player,f);
        end
        % find the best:
        jj = find(Qa == max(Qa));
        if numel(jj) > 1
            % choose randomly among equally good moves:
            jj = jj(randi(numel(jj),1,1));
        end
        move = moves(jj,:);
    end
end
