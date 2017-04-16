function [f_all w_all] = getFeatures

LR  = @fliplr;
UD  = @flipud;

% the features are encoded from the point of view of player == 1
f_generators = {
% "good for me":
[0 1 1 0],       1
[0 1 1 1 0],     10
[0 1 1 1 1 0],   90
[1 1 0 1 1],     10
[1 0 1 1 1],     20
[-1 1 1 1 1 0],  4
[1 1 1 1 1],     200
% "bad for me":
[0 -1 -1 0],       -1
[0 -1 -1 -1 0],    -70
[0 -1 -1 -1 -1 0], -90
[-1 0 -1 -1 -1],   -90
[-1 -1 0 -1 -1],   -90
[1 -1 -1 -1 -1 0], -80
[-1 -1 -1 -1 -1],  -100
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
    
    f_all = [f_all; fsym(:)];
    w_all = [w_all; w(n)*ones(numel(fsym),1)];
end
