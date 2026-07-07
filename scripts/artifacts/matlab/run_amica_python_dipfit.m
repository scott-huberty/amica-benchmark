datasets = 1:14; % setdiff(1:14, 10);
algorithms = [48 49]; % 48 = Py-EM, 49 = Py-DAAREM

results = struct( ...
    'dataset', {}, ...
    'algonum', {}, ...
    'ok', {}, ...
    'message', {});

result_idx = 0;
for alg_idx = 1:length(algorithms)
    ALGONUM = algorithms(alg_idx);
    for dataset_idx = 1:length(datasets)
        DATASET = datasets(dataset_idx);
        result_idx = result_idx + 1;
        results(result_idx).dataset = DATASET;
        results(result_idx).algonum = ALGONUM;
        results(result_idx).ok = false;
        results(result_idx).message = '';

        fprintf('Running DIPFIT for AMICA-Python: DATASET=%d ALGONUM=%d\n', DATASET, ALGONUM);
        try
            processdat;
            results(result_idx).ok = true;
            results(result_idx).message = 'ok';
        catch err
            results(result_idx).ok = false;
            results(result_idx).message = getReport(err, 'extended');
            fprintf(2, 'DIPFIT failed for DATASET=%d ALGONUM=%d\n%s\n', DATASET, ALGONUM, results(result_idx).message);
        end
    end
end

save('-mat', 'amica_python_dipfit_batch_results.mat', 'results');

if ~all([results.ok])
    error('One or more AMICA-Python DIPFIT runs failed. See amica_python_dipfit_batch_results.mat.');
end
