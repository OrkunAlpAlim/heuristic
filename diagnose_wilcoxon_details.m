clear; clc;

base_dir = 'results';
alg_a = 'fdb_se';
alg_b = 'se';
alpha = 0.05;

experiments = {'cec2022_10', 'cec2022_20'};

fprintf('%-12s %-5s %-12s %-12s %-12s %-10s %-5s\n', ...
    'Experiment', 'Func', 'median_A', 'median_B', 'medianDiff', 'p', 'sym');
fprintf('%s\n', repmat('-', 1, 78));

for ei = 1:numel(experiments)
    exp_name = experiments{ei};

    for func_num = 1:12
        [v_a, v_b] = load_paired_runs(base_dir, alg_a, alg_b, exp_name, func_num);

        if numel(v_a) < 2
            continue;
        end

        diffs = v_a - v_b;

        if all(diffs == 0)
            p = NaN;
            sym = '=';
        else
            try
                p = signrank(v_a, v_b);
            catch
                p = NaN;
            end

            direction = median(diffs);
            if direction == 0
                direction = mean(diffs);
            end

            if ~isnan(p) && p <= alpha && direction < 0
                sym = '+';
            elseif ~isnan(p) && p <= alpha && direction > 0
                sym = '-';
            else
                sym = '=';
            end
        end

        fprintf('%-12s F%-4d %-12.4e %-12.4e %-12.4e %-10.3g %-5s\n', ...
            exp_name, func_num, median(v_a), median(v_b), median(diffs), p, sym);
    end
end

function [v_a, v_b] = load_paired_runs(base_dir, alg_a, alg_b, exp_name, func_num)
    root_a = fullfile(base_dir, alg_a, exp_name, sprintf('F%d', func_num));
    root_b = fullfile(base_dir, alg_b, exp_name, sprintf('F%d', func_num));

    v_a = [];
    v_b = [];

    if ~isfolder(root_a) || ~isfolder(root_b)
        return;
    end

    runs_a = list_run_indices(root_a);
    runs_b = list_run_indices(root_b);
    common_runs = intersect(runs_a, runs_b);

    v_a = nan(numel(common_runs), 1);
    v_b = nan(numel(common_runs), 1);

    for i = 1:numel(common_runs)
        r = common_runs(i);
        v_a(i) = read_run_metric(fullfile(root_a, sprintf('run%d', r)));
        v_b(i) = read_run_metric(fullfile(root_b, sprintf('run%d', r)));
    end

    valid = ~isnan(v_a) & ~isnan(v_b);
    v_a = v_a(valid);
    v_b = v_b(valid);
end

function runs = list_run_indices(function_dir)
    runs = [];
    d = dir(function_dir);
    for k = 1:numel(d)
        if d(k).isdir
            tok = regexp(d(k).name, '^run(\d+)$', 'tokens', 'once');
            if ~isempty(tok)
                runs(end+1) = str2double(tok{1}); %#ok<AGROW>
            end
        end
    end
    runs = sort(runs);
end

function value = read_run_metric(run_dir)
    value = NaN;
    info_file = fullfile(run_dir, 'run_info.mat');
    if ~isfile(info_file)
        return;
    end

    S = load(info_file, 'run_info');
    if ~isfield(S, 'run_info')
        return;
    end

    ri = S.run_info;

    if isfield(ri, 'best_error') && isnumeric(ri.best_error) && ~isnan(ri.best_error)
        value = double(ri.best_error);
    elseif isfield(ri, 'best_fitness') && isnumeric(ri.best_fitness) && ~isnan(ri.best_fitness)
        value = double(ri.best_fitness);
    end
end