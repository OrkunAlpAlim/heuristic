% ----------------------------------------------------------------------- %
% FDB-guided Spherical Evolution (FDB-SE)
% Proposed variant for unconstrained benchmark problems
% Save this file as: proposed/fdb_se.m
% ----------------------------------------------------------------------- %
function [best_fitness, best_solution, curve, population_history, fitness_history] = fdb_se(problem)

    dim   = problem.dimension;
    lb    = problem.lb;
    ub    = problem.ub;
    maxFE = problem.maxFe;

    % Main parameters
    % q=25: slightly larger than the original q=20 to improve diversity,
    % but not so large that it consumes too much of the FE budget at startup.
    q = 25;
    if maxFE < q
        q = maxFE;
    end
    if q < 5
        error('FDB-SE requires maxFE >= 5 because the mutation uses at least five population members.');
    end

    FE = 0;
    curve = zeros(1, maxFE);

    history_size = 10000;
    sampling_interval = max(1, floor(maxFE / history_size));
    population_history = zeros(history_size, q, dim);
    fitness_history = zeros(history_size, q);
    history_index = 1;

    % Initialize population
    Positions = initialization(q, dim, ub, lb);
    [fitness, FE] = calculate_fitness(Positions', problem, FE);

    [global_best_fit, best_idx] = min(fitness);
    global_best_pos = Positions(best_idx, :);

    for eval_count = 1:min(FE, maxFE)
        curve(eval_count) = global_best_fit;
        [population_history, fitness_history, history_index] = record_history( ...
            eval_count, Positions, fitness, population_history, fitness_history, ...
            history_index, sampling_interval, history_size);
    end

    stall_counter = 0;

    while FE < maxFE

        progress = FE / maxFE;
        [~, y] = sort(fitness);

        % Rank-adaptive F: good individuals use smaller steps, weak individuals
        % keep larger exploratory steps. The whole interval shrinks over time.
        rankNorm = zeros(1, q);
        if q > 1
            rankNorm(y) = (0:q-1) / (q-1);  % best=0, worst=1
        end
        Fmax = 0.90 - 0.35 * progress;      % 0.90 -> 0.55
        Fmin = 0.35 - 0.15 * progress;      % 0.35 -> 0.20
        F_val = Fmin + (Fmax - Fmin) .* rankNorm + 0.05 * randn(1, q);
        F_val = min(max(F_val, 0.15), 0.95);

        % Adaptive subspace size. Early: wider coordinate updates.
        % Late: smaller coordinate updates for fine exploitation.
        if dim < 10
            pdMax = dim;
            pdMin = max(1, min(dim, 3));
        else
            pdMax = round((0.35 - 0.25 * progress) * dim);
            pdMax = min(dim, min(30, max(6, pdMax)));
            pdMin = min(pdMax, max(3, round(0.50 * pdMax)));
        end

        for j = 1:q
            if FE >= maxFE
                break;
            end

            kk = rand_excluding(q, j, 5);

            b = Positions(kk(1), :);
            f = Positions(kk(2), :);
            m = Positions(kk(3), :);
            n = Positions(kk(4), :);
            g = Positions(kk(5), :);
            h = Positions(j, :);

            guide_idx = select_fdb_guide(Positions, fitness, j);
            fdb_guide = Positions(guide_idx, :);

            paraDim = randi([pdMin, pdMax]);
            mm = randi(paraDim);
            mm = min(mm, dim);
            pp3 = sort(randperm(dim, mm));

            if length(pp3) == 1
                s1 = F_val(j) * HyperSphereTransform_1D(b, f, pp3);
                s2 = F_val(j) * HyperSphereTransform_1D(m, n, pp3);
            elseif length(pp3) == 2
                s1 = F_val(j) * HyperSphereTransform_2D(b, f, pp3);
                s2 = F_val(j) * HyperSphereTransform_2D(m, n, pp3);
            else
                s1 = F_val(j) * HyperSphereTransform(b, f, pp3);
                s2 = F_val(j) * HyperSphereTransform(m, n, pp3);
            end

            % Ensemble strategy schedule:
            %   early  : mostly rand-based exploration
            %   middle : FDB-guided balance
            %   late   : best-guided exploitation with small spherical noise
            r = rand;
            if progress < 0.35
                if r < 0.50
                    % SE/rand/1
                    h(pp3) = g(pp3) + s1;
                elseif r < 0.80
                    % moderated SE/rand/2
                    h(pp3) = g(pp3) + s1 + 0.50 * s2;
                else
                    % current-to-FDB/1
                    h(pp3) = h(pp3) + 0.45 * (fdb_guide(pp3) - h(pp3)) + s1;
                end
            elseif progress < 0.75
                if r < 0.70
                    % main FDB-guided balanced move
                    h(pp3) = h(pp3) + 0.65 * (fdb_guide(pp3) - h(pp3)) + 0.65 * s1;
                elseif r < 0.90
                    % best/1 exploitation, still with spherical perturbation
                    h(pp3) = global_best_pos(pp3) + 0.70 * s1;
                else
                    % current-to-best with a secondary difference vector
                    h(pp3) = h(pp3) + 0.50 * (global_best_pos(pp3) - h(pp3)) + 0.60 * s2;
                end
            else
                if r < 0.75
                    % late-stage intensification
                    h(pp3) = h(pp3) + 0.85 * (global_best_pos(pp3) - h(pp3)) + 0.25 * s1;
                else
                    % very local best-centered trial
                    h(pp3) = global_best_pos(pp3) + 0.35 * s1;
                end
            end

            temp = bound(h, ub, lb);

            old_global_best = global_best_fit;
            [temp_fit, FE] = calculate_fitness(temp', problem, FE);

            if temp_fit < fitness(j)
                Positions(j, :) = temp;
                fitness(j) = temp_fit;
            end

            if temp_fit < global_best_fit
                global_best_fit = temp_fit;
                global_best_pos = temp;
            end

            if global_best_fit < old_global_best
                stall_counter = 0;
            else
                stall_counter = stall_counter + 1;
            end

            if FE <= maxFE
                curve(FE) = global_best_fit;
                [population_history, fitness_history, history_index] = record_history( ...
                    FE, Positions, fitness, population_history, fitness_history, ...
                    history_index, sampling_interval, history_size);
            end

            % Stagnation pulse: when several consecutive trials do not improve
            % the global best, test a small Gaussian perturbation around best.
            if FE < maxFE && stall_counter >= 3 * q
                [~, yy] = sort(fitness);
                worst_idx = yy(end);

                trial = global_best_pos;
                localDim = max(1, round((0.12 - 0.08 * progress) * dim));
                localDim = min(dim, min(15, localDim));
                localSet = randperm(dim, localDim);
                range = get_range(dim, ub, lb);
                sigma = (0.01 + 0.04 * (1 - progress)) .* range(localSet);
                trial(localSet) = trial(localSet) + sigma .* randn(1, localDim);
                trial = bound(trial, ub, lb);

                [trial_fit, FE] = calculate_fitness(trial', problem, FE);

                if trial_fit < fitness(worst_idx)
                    Positions(worst_idx, :) = trial;
                    fitness(worst_idx) = trial_fit;
                end
                if trial_fit < global_best_fit
                    global_best_fit = trial_fit;
                    global_best_pos = trial;
                end

                stall_counter = 0;

                if FE <= maxFE
                    curve(FE) = global_best_fit;
                    [population_history, fitness_history, history_index] = record_history( ...
                        FE, Positions, fitness, population_history, fitness_history, ...
                        history_index, sampling_interval, history_size);
                end
            end
        end
    end

    for idx = 2:maxFE
        if curve(idx) == 0
            curve(idx) = curve(idx - 1);
        end
    end
    if maxFE >= 1 && curve(1) == 0
        curve(1) = global_best_fit;
    end

    best_fitness = global_best_fit;
    best_solution = global_best_pos;
end

%% --- FDB guide selection: high fitness quality + useful distance ---
function idx = select_fdb_guide(Positions, fitness, current_index)
    q = size(Positions, 1);

    fmin = min(fitness);
    fmax = max(fitness);
    if abs(fmax - fmin) < eps
        fitScore = ones(q, 1);
    else
        fitScore = (fmax - fitness(:)) ./ (fmax - fmin + eps); % minimization
    end

    current = Positions(current_index, :);
    dist = sqrt(sum((Positions - current) .^ 2, 2));
    dmax = max(dist);
    if dmax < eps
        distScore = zeros(q, 1);
    else
        distScore = dist ./ (dmax + eps);
    end

    score = 0.60 * fitScore + 0.40 * distScore;
    score(current_index) = -inf;

    [~, order] = sort(score, 'descend');
    topK = min(3, q - 1);
    idx = order(randi(topK));
end

%% --- Random indices excluding current individual ---
function idx = rand_excluding(n, exclude, k)
    pool = 1:n;
    pool(pool == exclude) = [];
    if numel(pool) >= k
        idx = pool(randperm(numel(pool), k));
    else
        idx = randi(n, 1, k);
    end
end

%% --- Hyper-Sphere Transform (General D-dimensional) ---
function ss = HyperSphereTransform(c, d, pp)
    D = length(pp);
    A = c(pp) - d(pp);
    R = norm(A, 2);

    O = zeros(1, D - 1);
    O(D - 1) = 2 * pi * rand;
    for i = 1:D - 2
        O(i) = rand * pi;
    end

    C = zeros(1, D);
    C(1) = R * prod(sin(O));
    for i = 2:D - 1
        C(i) = R * cos(O(i - 1)) * prod(sin(O(i:D - 1)));
    end
    C(D) = R * cos(O(D - 1));
    ss = C;
end

%% --- Hyper-Sphere Transform (1D) ---
function ss = HyperSphereTransform_1D(c, d, pp)
    R = abs(c(pp) - d(pp));
    ss = R * cos(2 * pi * rand);
end

%% --- Hyper-Sphere Transform (2D) ---
function ss = HyperSphereTransform_2D(c, d, pp)
    A = c(pp) - d(pp);
    R = norm(A, 2);
    o1 = 2 * pi * rand;
    C = zeros(1, 2);
    C(1) = R * sin(o1);
    C(2) = R * cos(o1);
    ss = C;
end

%% --- Initialization Function ---
function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    X = zeros(SearchAgents_no, dim);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        for i = 1:dim
            X(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
        end
    end
end

%% --- Boundary Handling: randomized reflection instead of hard clipping ---
function a = bound(a, ub, lb)
    if numel(ub) == 1
        ubv = repmat(ub, 1, numel(a));
        lbv = repmat(lb, 1, numel(a));
    else
        ubv = ub;
        lbv = lb;
    end

    range = ubv - lbv;

    upper = a > ubv;
    if any(upper)
        a(upper) = ubv(upper) - rand(1, sum(upper)) .* 0.25 .* range(upper);
    end

    lower = a < lbv;
    if any(lower)
        a(lower) = lbv(lower) + rand(1, sum(lower)) .* 0.25 .* range(lower);
    end

    a = min(max(a, lbv), ubv);
end

%% --- Bound range helper ---
function range = get_range(dim, ub, lb)
    if numel(ub) == 1
        range = repmat(ub - lb, 1, dim);
    else
        range = ub - lb;
    end
end
