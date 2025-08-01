function matlab_comprehensive_comparison_tests()
% MATLAB Comprehensive Comparison Tests for Katharsis
% ================================================
%
% This script runs comprehensive MATLAB/EEGLAB tests to generate ground truth
% data for comparison with Katharsis Backend functionality.
%
% Tests included:
% 1. Signal filtering (high-pass, low-pass, band-pass, notch)
% 2. Re-referencing (average, common reference)
% 3. ICA analysis with artifact detection
% 4. Event detection and epoching
% 5. ERP analysis
% 6. Edge case handling
%
% Requires: MATLAB with Signal Processing Toolbox and EEGLAB
%
% Author: porfanid
% Version: 1.0 - Comprehensive Ground Truth Generation

    fprintf('🔧 Starting MATLAB Comprehensive Comparison Tests...\n');
    
    % Check if EEGLAB is available
    if ~exist('eeglab', 'file')
        warning('EEGLAB not found in path. Some tests may fail.');
        fprintf('Please add EEGLAB to MATLAB path or install it.\n');
    else
        % Initialize EEGLAB
        eeglab('nogui');
        fprintf('✅ EEGLAB initialized successfully\n');
    end
    
    % Test data directory
    test_data_dir = 'tests/matlab_edge_case_data/';
    results_dir = 'tests/matlab_ground_truth_results/';
    
    % Create results directory
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    
    % Run all test categories
    try
        % 1. Basic signal processing tests
        fprintf('\n📊 Running signal processing ground truth tests...\n');
        run_signal_processing_tests(test_data_dir, results_dir);
        
        % 2. ICA analysis tests
        fprintf('\n🧠 Running ICA analysis ground truth tests...\n');
        run_ica_analysis_tests(test_data_dir, results_dir);
        
        % 3. Event detection and epoching tests
        fprintf('\n⏱️  Running event detection ground truth tests...\n');
        run_event_detection_tests(test_data_dir, results_dir);
        
        % 4. Edge case validation tests
        fprintf('\n🚨 Running edge case validation tests...\n');
        run_edge_case_validation_tests(test_data_dir, results_dir);
        
        % 5. Generate comprehensive comparison report
        fprintf('\n📄 Generating comprehensive comparison report...\n');
        generate_comparison_report(results_dir);
        
        fprintf('\n🎯 All MATLAB ground truth tests completed successfully!\n');
        fprintf('Results saved in: %s\n', results_dir);
        
    catch ME
        fprintf('❌ MATLAB tests failed with error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
end

function run_signal_processing_tests(test_data_dir, results_dir)
% Run signal processing ground truth tests
    
    % Test cases to process
    test_cases = {'normal_5ch', 'normal_10ch', 'high_noise', 'with_artifacts', 'low_noise'};
    
    for i = 1:length(test_cases)
        test_name = test_cases{i};
        fprintf('  Processing %s...\n', test_name);
        
        try
            % Load test data
            data_file = fullfile(test_data_dir, [test_name '.mat']);
            if ~exist(data_file, 'file')
                fprintf('    ⚠️  Test data file not found: %s\n', data_file);
                continue;
            end
            
            load(data_file);
            
            % Convert to EEGLAB format
            EEG = create_eeglab_structure(data, ch_names, sfreq);
            
            % Test 1: High-pass filtering (1 Hz)
            EEG_hp = pop_eegfiltnew(EEG, 'locutoff', 1);
            hp_filtered_data = EEG_hp.data;
            
            % Test 2: Low-pass filtering (40 Hz)
            EEG_lp = pop_eegfiltnew(EEG, 'hicutoff', 40);
            lp_filtered_data = EEG_lp.data;
            
            % Test 3: Band-pass filtering (1-40 Hz)
            EEG_bp = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);
            bp_filtered_data = EEG_bp.data;
            
            % Test 4: Notch filtering (50 Hz)
            EEG_notch = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);
            notch_filtered_data = EEG_notch.data;
            
            % Test 5: Average re-referencing
            EEG_avg_ref = pop_reref(EEG, []);
            avg_ref_data = EEG_avg_ref.data;
            
            % Test 6: Common reference (first channel)
            EEG_common_ref = pop_reref(EEG, 1);
            common_ref_data = EEG_common_ref.data;
            
            % Save ground truth results
            ground_truth = struct();
            ground_truth.original_data = data;
            ground_truth.hp_filtered_1hz = hp_filtered_data;
            ground_truth.lp_filtered_40hz = lp_filtered_data;
            ground_truth.bp_filtered_1_40hz = bp_filtered_data;
            ground_truth.notch_filtered_50hz = notch_filtered_data;
            ground_truth.avg_referenced = avg_ref_data;
            ground_truth.common_referenced = common_ref_data;
            ground_truth.test_parameters = struct( ...
                'sfreq', sfreq, ...
                'n_channels', length(ch_names), ...
                'n_samples', size(data, 2), ...
                'ch_names', {ch_names} ...
            );
            
            % Save results
            result_file = fullfile(results_dir, [test_name '_signal_processing_ground_truth.mat']);
            save(result_file, 'ground_truth');
            
            fprintf('    ✅ Signal processing ground truth saved\n');
            
        catch ME
            fprintf('    ❌ Failed to process %s: %s\n', test_name, ME.message);
        end
    end
end

function run_ica_analysis_tests(test_data_dir, results_dir)
% Run ICA analysis ground truth tests
    
    test_cases = {'normal_5ch', 'normal_10ch', 'high_noise', 'with_artifacts'};
    
    for i = 1:length(test_cases)
        test_name = test_cases{i};
        fprintf('  Processing ICA for %s...\n', test_name);
        
        try
            % Load test data
            data_file = fullfile(test_data_dir, [test_name '.mat']);
            if ~exist(data_file, 'file')
                continue;
            end
            
            load(data_file);
            
            % Convert to EEGLAB format and preprocess
            EEG = create_eeglab_structure(data, ch_names, sfreq);
            
            % Apply basic preprocessing for ICA
            EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 40);
            
            % Run ICA (using EEGLAB's default - extended infomax)
            EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1);
            
            % Extract ICA results
            ica_weights = EEG.icaweights;
            ica_sphere = EEG.icasphere;
            ica_activations = EEG.icaact;
            mixing_matrix = pinv(ica_weights * ica_sphere);
            n_components = size(ica_weights, 1);
            
            % Basic artifact detection (simplified)
            artifact_components = detect_artifacts_simple(EEG);
            
            % Save ICA ground truth
            ica_ground_truth = struct();
            ica_ground_truth.weights = ica_weights;
            ica_ground_truth.sphere = ica_sphere;
            ica_ground_truth.activations = ica_activations;
            ica_ground_truth.mixing_matrix = mixing_matrix;
            ica_ground_truth.n_components = n_components;
            ica_ground_truth.artifact_components = artifact_components;
            ica_ground_truth.preprocessed_data = EEG.data;
            
            % Save results
            result_file = fullfile(results_dir, [test_name '_ica_ground_truth.mat']);
            save(result_file, 'ica_ground_truth');
            
            fprintf('    ✅ ICA ground truth saved (%d components)\n', n_components);
            
        catch ME
            fprintf('    ❌ ICA failed for %s: %s\n', test_name, ME.message);
        end
    end
end

function run_event_detection_tests(test_data_dir, results_dir)
% Run event detection and epoching ground truth tests
    
    test_cases = {'normal_5ch', 'normal_10ch'};
    
    for i = 1:length(test_cases)
        test_name = test_cases{i};
        fprintf('  Processing events for %s...\n', test_name);
        
        try
            % Load test data
            data_file = fullfile(test_data_dir, [test_name '.mat']);
            if ~exist(data_file, 'file')
                continue;
            end
            
            load(data_file);
            
            % Create EEGLAB structure with synthetic events
            EEG = create_eeglab_structure(data, ch_names, sfreq);
            
            % Create synthetic events for testing
            n_events = 20;
            event_latencies = sort(randperm(size(data, 2) - 500, n_events)) + 250;
            event_types = randi(3, 1, n_events);
            
            % Add events to EEG structure
            for j = 1:n_events
                EEG.event(j).latency = event_latencies(j);
                EEG.event(j).type = num2str(event_types(j));
            end
            EEG.urevent = EEG.event;
            
            % Extract epochs around events
            EEG_epochs = pop_epoch(EEG, {'1', '2', '3'}, [-0.2 0.8]);
            
            % Calculate ERP (average across epochs)
            erp_data = mean(EEG_epochs.data, 3);
            
            % Event detection ground truth
            event_ground_truth = struct();
            event_ground_truth.events = EEG.event;
            event_ground_truth.event_latencies = [EEG.event.latency];
            event_ground_truth.event_types = {EEG.event.type};
            event_ground_truth.epochs = EEG_epochs.data;
            event_ground_truth.erp = erp_data;
            event_ground_truth.epoch_times = EEG_epochs.times;
            event_ground_truth.n_epochs = EEG_epochs.trials;
            
            % Save results
            result_file = fullfile(results_dir, [test_name '_events_ground_truth.mat']);
            save(result_file, 'event_ground_truth');
            
            fprintf('    ✅ Event detection ground truth saved (%d events, %d epochs)\n', ...
                n_events, EEG_epochs.trials);
            
        catch ME
            fprintf('    ❌ Event processing failed for %s: %s\n', test_name, ME.message);
        end
    end
end

function run_edge_case_validation_tests(test_data_dir, results_dir)
% Run edge case validation tests
    
    fprintf('  Testing edge case scenarios...\n');
    
    edge_cases = struct();
    
    % Test 1: Single channel data
    single_ch_data = randn(1, 2500);  % 10 seconds at 250 Hz
    edge_cases.single_channel = struct( ...
        'data', single_ch_data, ...
        'expected_ica_fail', true, ...
        'expected_message', 'Insufficient channels for ICA' ...
    );
    
    % Test 2: Data with NaN values
    nan_data = randn(5, 2500);
    nan_indices = randperm(numel(nan_data), 100);
    nan_data(nan_indices) = NaN;
    edge_cases.nan_data = struct( ...
        'data', nan_data, ...
        'expected_preprocessing_fail', true, ...
        'expected_message', 'Data contains NaN values' ...
    );
    
    % Test 3: Data with infinite values
    inf_data = randn(5, 2500);
    inf_indices = randperm(numel(inf_data), 10);
    inf_data(inf_indices) = Inf;
    edge_cases.inf_data = struct( ...
        'data', inf_data, ...
        'expected_preprocessing_fail', true, ...
        'expected_message', 'Data contains infinite values' ...
    );
    
    % Test 4: Very short data (insufficient for ICA)
    short_data = randn(5, 250);  % 1 second
    edge_cases.short_data = struct( ...
        'data', short_data, ...
        'expected_ica_fail', true, ...
        'expected_message', 'Insufficient data duration for ICA' ...
    );
    
    % Test 5: Zero variance data
    zero_var_data = randn(5, 2500);
    zero_var_data(1, :) = 0;  % First channel has zero variance
    edge_cases.zero_variance = struct( ...
        'data', zero_var_data, ...
        'expected_ica_warning', true, ...
        'expected_message', 'Low variance channel detected' ...
    );
    
    % Save edge cases
    result_file = fullfile(results_dir, 'edge_cases_ground_truth.mat');
    save(result_file, 'edge_cases');
    
    fprintf('    ✅ Edge case validation data saved\n');
end

function generate_comparison_report(results_dir)
% Generate comprehensive comparison report
    
    report_file = fullfile(results_dir, 'matlab_ground_truth_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, 'MATLAB Ground Truth Generation Report\n');
    fprintf(fid, '====================================\n\n');
    fprintf(fid, 'Generated on: %s\n', datestr(now));
    fprintf(fid, 'MATLAB Version: %s\n', version);
    
    % Check if EEGLAB is available
    if exist('eeglab', 'file')
        fprintf(fid, 'EEGLAB Version: Available\n');
    else
        fprintf(fid, 'EEGLAB Version: Not Available\n');
    end
    
    fprintf(fid, '\nGenerated Files:\n');
    fprintf(fid, '---------------\n');
    
    % List all generated files
    mat_files = dir(fullfile(results_dir, '*.mat'));
    for i = 1:length(mat_files)
        fprintf(fid, '- %s\n', mat_files(i).name);
    end
    
    fprintf(fid, '\nTest Categories:\n');
    fprintf(fid, '---------------\n');
    fprintf(fid, '1. Signal Processing (filtering, re-referencing)\n');
    fprintf(fid, '2. ICA Analysis (component extraction, artifact detection)\n');
    fprintf(fid, '3. Event Detection (epoching, ERP computation)\n');
    fprintf(fid, '4. Edge Case Validation (error conditions)\n');
    
    fprintf(fid, '\nUsage Instructions:\n');
    fprintf(fid, '------------------\n');
    fprintf(fid, '1. Load ground truth data in Python using scipy.io.loadmat\n');
    fprintf(fid, '2. Compare Katharsis results with ground truth using similarity metrics\n');
    fprintf(fid, '3. Use correlation, RMSE, or other metrics for validation\n');
    fprintf(fid, '4. Ensure edge cases are handled appropriately\n');
    
    fclose(fid);
    
    fprintf('📄 Comparison report saved to: %s\n', report_file);
end

function EEG = create_eeglab_structure(data, ch_names, sfreq)
% Create EEGLAB EEG structure from data
    
    EEG = eeg_emptyset();
    EEG.data = data;
    EEG.srate = sfreq;
    EEG.nbchan = size(data, 1);
    EEG.pnts = size(data, 2);
    EEG.trials = 1;
    EEG.xmin = 0;
    EEG.xmax = (size(data, 2) - 1) / sfreq;
    
    % Set channel names
    for i = 1:length(ch_names)
        EEG.chanlocs(i).labels = ch_names{i};
    end
    
    % Check consistency
    EEG = eeg_checkset(EEG);
end

function artifacts = detect_artifacts_simple(EEG)
% Simple artifact detection for ground truth
    
    artifacts = [];
    
    if isempty(EEG.icaweights)
        return;
    end
    
    % Simple heuristics for artifact detection
    n_components = size(EEG.icaweights, 1);
    
    for comp = 1:n_components
        % Get component activation
        if ~isempty(EEG.icaact)
            comp_activation = EEG.icaact(comp, :);
        else
            comp_activation = EEG.icaweights(comp, :) * EEG.icasphere * EEG.data;
        end
        
        % Simple metrics for artifact detection
        kurt_val = kurtosis(comp_activation);
        var_val = var(comp_activation);
        
        % Classify as artifact if high kurtosis (spiky) or very high/low variance
        if kurt_val > 5 || var_val > 10 * median(var(EEG.icaact, [], 2)) || var_val < 0.1 * median(var(EEG.icaact, [], 2))
            artifacts = [artifacts, comp];
        end
    end
end