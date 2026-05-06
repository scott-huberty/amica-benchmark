datasets = 1:14; % setdiff(1:14, 10);
ALGONUM = 48;

results = struct( ...
    'dataset', num2cell(datasets), ...
    'ok', num2cell(false(size(datasets))), ...
    'message', repmat({''}, size(datasets)));

for idx = 1:length(datasets)
    DATASET = datasets(idx);
    fprintf('Running DIPFIT for AMICA-Python: DATASET=%d ALGONUM=%d\n', DATASET, ALGONUM);
    try
        processdat;
        results(idx).ok = true;
        results(idx).message = 'ok';
    catch err
        results(idx).ok = false;
        results(idx).message = getReport(err, 'extended');
        fprintf(2, 'DIPFIT failed for DATASET=%d ALGONUM=%d\n%s\n', DATASET, ALGONUM, results(idx).message);
    end
end

save('-mat', 'amica_python_dipfit_batch_results.mat', 'results');

if ~all([results.ok])
    error('One or more AMICA-Python DIPFIT runs failed. See amica_python_dipfit_batch_results.mat.');
end
