function fit_result = fit_hybrid_bias_models(data, savedatapath)

%filename = regexp(fileName, '(?<=data4model).*', 'match');
savefilename = savedatapath;

if ~exist(savefilename)
    % read the data prepared before, fit the hybrid model on
    %fileName = 'Z:\HongliWang\Juvi_ASD Deterministic\TSC2\Summary\Results\data4modelAB-AB1.mat';
    
    % load(fileName);
    % subjects = unique(X.Animal_ID);
    % genotypes = cell(length(subjects),1);

    %% load the estimated Q value from previous session if needed
    % if contains(filename,'AB2') 
    %     loadfile = 'AB-AB1';
    % elseif contains(filename,'AB3')
    %     loadfile = 'AB-AB2';
    % elseif contains(filename, 'CD2')
    %     loadfile = 'CD-CD1';
    % elseif contains(filename,'CD3')
    %     loadfile = 'CD-CD2';
    % else
    %     loadfile = nan;
    % end
    % learntQ = zeros(2,2,length(subjects))+0.5;

    beta_mu = 5;
    beta_sigma = 7;

    %% Define priors of model parameters
    % Define functions to sample model parameters from uniform distributions
    beta_sample = @(x) beta(5, 7); % Sampling function for beta parameter
    alpha_sample = @(x) unifrnd(0, 1); % Sampling function for alpha parameter
    forget_sample = @(x) unifrnd(0, 1); % Sampling function for forget parameter
    stick_sample = @(x) unifrnd(-1, 1); % Sampling function for stickiness parameter
    stick_nr_sample = @(x) unifrnd(-20, 20);
    alpha_CK_sample = @(x) unifrnd(0,1);
    beta_CK_sample = @(x) normrnd(0,7);
    bias_sample = @(x) unifrnd(0, 1); % Sampling function for bias parameter
    w_sample = @(x) unifrnd(0, 1); % Sampling function for w parameter
    eps_sample = @(x) unifrnd(0, 1); % Sampling function for epsilon parameter
    lapse_sample = @(x) unifrnd(0, 1); % Sampling function for lapse parameter (T_1^0)
    rec_sample = @(x) unifrnd(0, 1); % Sampling function for recover parameter (T_0^1)
    sigma_sample = @(x) unifrnd(0,1); % sampling function for noise parameter
    forget_sample = @(x) unifrnd(0,1);
    %% set params
    M1s = [];


    % RL4s_hybrid


    curr_model = [];
    curr_model.name = 'a0b1s_hybrid';
    curr_model.pMin = [1e-6 1e-6 -1 1e-6 1e-6 1e-6];
    curr_model.pMax = [20     1   1   1    1    1];
    curr_model.pdfs = {beta_sample, alpha_sample, stick_sample, lapse_sample, rec_sample, bias_sample}; % Sampling functions for model parameters
    curr_model.pnames = {'beta','alpha+','s1','lapse','rec','bias'};

    M1s{1}=curr_model;


    
    % for forgetting model
    M2s = [];
    curr_model = [];
    curr_model.name = 'a0b1s_hybrid';
    curr_model.pMin = [1e-6 1e-6 -1 1e-6 1e-6 1e-6 1e-6];
    curr_model.pMax = [20     1   1   1    1    1    1 ];
    curr_model.pdfs = {beta_sample, alpha_sample, stick_sample, lapse_sample, rec_sample, bias_sample, forget_sample}; % Sampling functions for model parameters
    curr_model.pnames = {'beta','alpha+','s1','lapse','rec','bias','forget'};

    M2s{1}=curr_model;


    %% Fit models
    % protocolName = regexp(fileName, '(?<=data4model).*', 'match');
    % if any(contains(protocolName{1}, {'CD1','CD2','CD3'}))
    %     X.schedule(X.schedule<=2) = nan;
    %     X.schedule = X.schedule-2;
    % elseif any(contains(protocolName{1}, {'DC1','DC2','DC3', 'DC4', 'DC5'}))
    %     X.schedule(X.schedule<=4) = nan;
    %     X.schedule = X.schedule-4;
    % end

    for sess = 1
        %if isnan(loadfile)
            Ms = M1s;
        %else
        %    Ms = M2s;
        %end
        All_Params = cell(length(Ms), 1);
        All_fits = cell(length(Ms), 1);
        for m = 1:length(Ms)
            fit_model = Ms{m};
            pmin = fit_model.pMin;
            pmax = fit_model.pMax;
            pdfs = fit_model.pdfs;

            fitmeasures = cell(length(subjects), 1);
            fitparams = cell(length(subjects), 1);

            for k = 1:length(subjects) % no parallel processing
                %parfor k = 1:length(subjects) % parallel processing
                tempgeno = unique(X.genotype(X.Animal_ID==subjects(k)));
                genotypes{k} = tempgeno{1};
                s = subjects(k);
                T = find(X.Animal_ID==s);
                % if CD

                temp_data = [X.schedule(T) X.action(T) X.reward1(T)>0];
                temp_data = temp_data(~isnan(temp_data(:,2))&~isnan(temp_data(:,1)),:);
                this_data.s = temp_data(:,1);
                this_data.c = temp_data(:,2);
                this_data.r = temp_data(:,3);
                this_data.Q = learntQ(:,:,k);
                % Sample parameter starting values
                par = zeros(length(pmin), 1);
                for p_ind = 1:length(pmin)
                    par(p_ind) = pdfs{p_ind}(0); % 0 is the random seed
                end


                % Define the objective function for optimization
                llhfun = @(p) feval([fit_model.name, '_llh'], p, this_data);
                if sum(strcmp(fit_model.pnames, 'beta')) > 0
                    beta_idx = strcmp(fit_model.pnames, 'beta');
                    myfitfun = @(p) llhfun(p) + sum((p(beta_idx) - beta_mu).^2 ./ (2*beta_sigma.^2));
                else
                    myfitfun = @(p) llhfun(p);
                end
                rng default % For reproducibility
                fmincon_opts = optimoptions(@fmincon, 'Algorithm', 'sqp');
                problem = createOptimProblem('fmincon', 'objective', myfitfun, 'x0', par, 'lb', pmin, 'ub', pmax, 'options', fmincon_opts);
                gs = GlobalSearch;
                [param, llh] = run(gs, problem);

                % Calculate fit measures (AIC, BIC, etc.)
                ntrials = size(this_data.s,1);
                AIC = 2 * llh + 2 * length(param);
                BIC = 2 * llh + log(ntrials) * length(param);
                AIC0 = -2 * log(1/3) * ntrials;
                psr2 = (AIC0 - AIC) / AIC0;

                % Store fit measures and parameters for each subject
                fitmeasures{k} = [k llh AIC BIC psr2 AIC0];
                fitparams{k} = param';
            end

            % Store fit measures and parameters for each model
            All_Params{m} = cell2mat(fitparams);
            All_fits{m} = cell2mat(fitmeasures);
        end

        % Reformat All_fits matrix
        temp = All_fits;
        tempParam = All_Params;
        All_params = cell(length(Ms),1);
        All_fits = zeros(length(subjects), size(temp{1}, 2), length(Ms));

        for i = 1:length(Ms)
            All_fits(:, :, i) = temp{i};
            All_params{i} = tempParam{i};
        end
        %params = All_Params{1};


        fit_result.All_fits = All_fits;
        fit_result.All_params = All_params;
        fit_result.subjects = subjects;
        fit_result.genotypes = genotypes;
        save(savefilename, 'All_fits','All_params', 'subjects', 'genotypes')
    end

