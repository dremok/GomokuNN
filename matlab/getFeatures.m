function [f w] = getFeatures

LR  = @fliplr;
UD  = @flipud;

% the features are encoded from the point of view of player == 1
f_generators = {
% "good for me":
[0 1 1 0],      0.1360
[0 1 1 1 0],    0.2662
[0 1 1 1 1 0],  0.9232
[0 1 0 1 1 0],  0.1967
[1 1 1 1 1],    1.5843
% "bad for me":
[0 -1 -1 0],        -0.1594
[0 -1 -1 -1 0],     -0.6747
[0 -1 -1 -1 -1 0],  -1.3054
[0 -1 -1 0 -1 0],   -1.3054
[-1 -1 0 -1 -1],    -0.7254
[-1 -1 -1 0 -1],    -0.5805
[1 -1 -1 -1 -1 0],  -1.1453
};

w = cell2mat(f_generators(:,2));
f_generators = f_generators(:,1);

%% 1d-features -> diagonal versions

f_diag = {};
w_diag = [];
for n = 1:numel(f_generators)
    if any(size(f_generators{n}) == 1)
        f_ = 2*ones(numel(f_generators{n}));
        f_ = f_ - diag(diag(f_)) + diag(f_generators{n}); 
        f_diag{end + 1} = f_;
        w_diag(end + 1) = w(n);
    end
end

f_generators = [f_generators; f_diag(:)];
w            = [w; w_diag(:)];

%% symmetrical versions of each generator
    
f_all = {};
w_all = [];
f_group = [];
for n = 1:numel(f_generators)
    f = f_generators{n};
    
    % all symmetries of f:
    fsymAll = cell(8,1);
    fsymAll{1} = f;
    fsymAll{2} = LR(f);
    fsymAll{3} = UD(f);
    fsymAll{4} = LR(UD(f));
    fsymAll{5} = f';
    fsymAll{6} = LR(f');
    fsymAll{7} = UD(f');
    fsymAll{8} = LR(UD(f'));

    % collect the non-reduntant ones:
    fsym = {};
    for j = 1:numel(fsymAll)
        isRedundant = 0;
        for k = 1:numel(fsym)
            if all(size(fsym{k}) == size(fsymAll{j}))
                if all(fsym{k}(:) == fsymAll{j}(:))
                    isRedundant = 1;
                end
            end
        end
        if ~isRedundant
            fsym{end + 1} = fsymAll{j};
        end
    end
    
    f_all   = [f_all; fsym(:)];
    w_all   = [w_all; w(n)*ones(numel(fsym),1)];
    f_group = [f_group; n*ones(numel(fsym),1)];
end

%% "encoded" version (i.e. ready for convolutions)

for j = 1:size(f_all,1)
    f_ = f_all{j} + 2;
    [nx, ny] = size(f_);
    if nx > 1, f_ = flipud(f_); end
    if ny > 1, f_ = fliplr(f_); end
    f_all{j,2} = 10.^f_.*(f_ ~= 4);
    
    f_all{j,3} = f_all{j,2}(:)'*f_(:);
end

%% output

clear f
f.pattern = f_all;
f.group   = f_group;

w = NaN(max(f.group),1);
for j = 1:max(f.group)
    k = find(f.group == j,1);
    w(j) = w_all(k);
end
