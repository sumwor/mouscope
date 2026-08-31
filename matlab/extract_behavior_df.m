function [result] = extract_behavior_df(filename)
    %% The function takes in filename of an exper structure and outputs behavioral dataframe (matlab table), and save into .csv for python processing
    % filename: string
    % outpath: string, path where csv is going to be saved
    % Returns:
    %   out: table of dataframe, also saved as csv in `outpath`

    % loading data
    data = get_RLWM_EventTimes(filename);
    dmat = data.RLWM_EventTimes';
    
    %% Getting basic event time features
    % identify outcome events, check # of trials = # outcome evnts
    outcome_inds = find(dmat(:, 1) > 80);
    out.trial = (1:length(outcome_inds))';
    out.outcome = dmat(outcome_inds, 2);
    odor_inds = find(floor(dmat(:, 1)) == 7);
    
    % look backward in time and identify all center_in/outs
    % center_in
    ci_times = backward_times(dmat, outcome_inds, @(region) (region(:, 1) == 1));
    out.center_in = ci_times;
   
    % center_out
    co_times = backward_times(dmat, outcome_inds, @(region) (region(:, 1) == 2));
    out.center_out = co_times; 
    
    % side_in
    si_times = backward_times(dmat, outcome_inds, @(region) ((region(:, 1) == 3) | (region(:, 1) == 5)));
    % go over every trial, is si_times is smaller than ci_times, it is in
    % fact a miss trial, set the value to be NaN
    si_times(si_times-ci_times<0)=NaN;
    out.side_in = si_times;
    
    % look forward in time with trial# k + 0.5 and find last_side_out
    so_times = NaN(length(outcome_inds), 1);
    for i=1:length(outcome_inds)
        start_ind = outcome_inds(i);
        if i == length(outcome_inds)
            end_ind = length(dmat);
        else
            end_ind = odor_inds(i+1);
        end
        region = dmat(start_ind:end_ind, :);
        so_time = region((region(:, 1) == 44) | (region(:, 1) == 66), 2);
        if ~isempty(so_time)
            so_times(i) = so_time(end);
        end
    end
    out.last_side_out = so_times;
       
    %% get task features 
    % choice side
    trial_sel = ismember(dmat(:, 2), si_times) & (dmat(:, 1) < 80);
    choice_trials = uint16(dmat(trial_sel, 3));
    actions = NaN(length(outcome_inds), 1);
    actions(choice_trials) = (dmat(trial_sel, 1) - 3) / 2;
    out.actions = actions;
    
    % # water valve open
    waters = NaN(length(outcome_inds), 1);
    water_sel = floor(dmat(:, 1)) == 9;
    water_given = uint16(dmat(water_sel, 3));
    waters(water_given) = mod(dmat(water_sel, 1), 1) * 100;
    out.reward = waters;
    
    %get trial types
    trial_types = mod(dmat(floor(dmat(:,1)) > 80, 1), 1) / 10;
    out.trial_types = trial_types;
    
    % odor identity (7.XX)
    odors = mod(dmat(floor(dmat(:, 1)) == 7, 1), 1) * 100;
    out.odors=odors;
    
    % additional features
    out.port_side = data.portside';
    out.schedule = data.schedule';
    out.odor_name = data.odor_name';
    out.odor_dur = data.odor_dur';
    
    % need the absolute start time to align between boxes
    out.start_time = ones(length(out.port_side),1)*data.startTime;
    result = struct2table(out); % "AsArray", true because fields have different numbers of rows?
end


function [result] = backward_times(dmat, outcome_inds, region_func)
    result = NaN(length(outcome_inds), 1);
    for i=1:length(outcome_inds)
        if i == 1
            start_ind = 1;
        else
            start_ind = outcome_inds(i-1);
        end
        end_ind = outcome_inds(i);
        region = dmat(start_ind:end_ind, :);
        itime = region(region_func(region), 2);
        if ~isempty(itime)
            result(i) = itime(end);
        end
    end
end