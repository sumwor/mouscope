function out=get_RLWM_EventTimes(varargin)
% out=get_RLWM_EventTimes(exper_file_name)
% get_RLWM_EventTimes is a function that reads the odor_RLWM data file
% and output a structure variable "out"
% (1) out.GoNG_EventTimes is in three rows: [eventID; eventTime; trial].
% List of eventID for the first row of out.GoNG_EventTimes
% eventID=1:   center port in
% eventID=2:   center port out
% eventID=3:   left port in
% eventID=4:   left port out
% eventID=44:  Last left port out
% eventID=5:   right port 1n
% eventID=6:   right port out
% eventID=66:  Last right port out
% eventID=7.01:  new trial, odor 1 ON
% eventID=7.02:  new trial, odor 2 ON
% eventID=7.0n:  new trial, odor n ON
% eventID=7.16:  new trial, odor 16 ON
% eventID=81.0: Correct response, withdraw too early
% eventID=81.1: Correct response, 1 drops rewarded (valve on) *NONE*
% eventID=81.2: Correct response, 2 drops rewarded (valve on)
% eventID=81.3: Correct response, 3 drops rewarded (valve on)
% eventID=82:    False Go (lick), white noise on  ???
% eventID=83:    Missed to respond
% eventID=84:    Aborted outcome
% eventID=9.01:  Water Valve on 1 time (1 reward)
% eventID=9.02:  Water Valve on 2 times (2 rewards)
% eventID=9.03:  Water Valve on 3 times (3 rewards)
% (2) out.odor_name is the name of the odor referenced in eventID=7.x (in
%     ASCII code, use char(out.odor_name) to get the alphabet name
% (3) out.odor_dur is the duration of the odor
% (4) out.schedule is the stimulus schedule for each trial
% (5) out.portside is the port_side schedule (-1:NoGo, 0:probe, 2:Go/left)
% (6) out.result is the result for each trial
% 01/31/2022 Lung-Hao Tai

% trial based, not time based: set size?

warning('off','MATLAB:timer:incompatibleTimerLoad');
warning('off','MATLAB:unknownElementsNowStruc');
out = [];
if nargin ==1
    arg = varargin{1};
    if ischar(arg) || iscellstr(arg) || isstring(arg)
        filename=arg;
        full_filename=which(filename);
        if isempty(full_filename)
            full_filename=filename;
        end
        dr=dir(full_filename);
        if ~isempty(dr)
            data=load(full_filename);
        else
            data = [];
        end
    elseif isfield(arg, 'exper')
        data = arg;
    else
        data = [];
    end
else
    disp('Please specify an exper filename in string');
    eval('help get_RLWM_EventTimes');
end

RLWM_EventTimes=[];
RLWM_EventTimes_n=0;
kk=0;

if isfield(data.exper,'odor_rlwm_automatic') & ~isfield(data.exper, 'odor_rlwm')
    useField = 'odor_rlwm_automatic';
elseif isfield(data.exper, 'odor_rlwm') & ~isfield(data.exper,'odor_rlwm_automatic')
    useField = 'odor_rlwm';
else
    % if both exist
    CountedTrial_1=data.exper.odor_rlwm.param.countedtrial.value;
    CountedTrial_2=data.exper.odor_rlwm_automatic.param.countedtrial.value;
    if CountedTrial_1 > 0 & CountedTrial_2 == 0
        useField = 'odor_rlwm';
    elseif CountedTrial_2 > 0 & CountedTrial_1 == 0
        useField = 'odor_rlwm_automatic';
    elseif CountedTrial_1 > 0 & CountedTrial_2 > 0
        useField = 'odor_rlwm_automatic';
    end
end

if ~isempty(data)
    trial_events=data.exper.rpbox.param.trial_events.value;
    if isfield(data.exper,useField)
        CountedTrial=data.exper.(useField).param.countedtrial.value;
        Result=data.exper.(useField).param.result.value(1:CountedTrial);
        portside=data.exper.(useField).param.port_side.value(1:CountedTrial);
        schedule=data.exper.(useField).param.schedule.value(1:CountedTrial);
        OdorChannelSchedule=data.exper.(useField).param.odorchannel.value(1:CountedTrial);
        OdorName=data.exper.(useField).param.odorname.value(1:CountedTrial);
%         ITI=data.exper.odor_rlwm.param.iti.trial(1:CountedTrial);
%         lwatervalvedur=data.exper.odor_rlwm.param.lwatervalvedur.value;
%         rwatervalvedur=data.exper.odor_rlwm.param.rwatervalvedur.value;
%         boxrig=data.exper.control.param.expid.value;
        protocol='odor_rlwm';
        StimParam=data.exper.(useField).param.stimparam.value;
        param_string=data.exper.(useField).param.stimparam.user;
        LeftP  =str2double(StimParam(:,strcmp(param_string,'left reward ratio')));
        RightP =str2double(StimParam(:,strcmp(param_string,'right reward ratio')));
        LeftRewardP=LeftP(schedule);
        RightRewardP=RightP(schedule);
    else
        error('no Odor_RLWM session found');
        return;
    end

    % start trials based sorting of events
    valid_trials=logical(size(Result).*0);
    for k=1:CountedTrial
        if k==1
            tt1=0;
            try
                if ismember(Result(k),[1.2, 1.3]) % two or three drop H2O
                    tt2=data.exper.(useField).param.trial_events.trial{k}(end,3);
                    kk=kk+1;
                else
                    tt2=data.exper.(useField).param.trial_events.trial{k}(:,3);
                    if length(tt2)>1
                        tt2=tt2(1);
                    end
                    kk=kk+1;
                end
            catch
                tt2=0;
            end
        else
            tt1=tt2;
            if ~isempty(data.exper.(useField).param.trial_events.trial{k})
                if ismember(Result(k),[1.2, 1.3]) % two or three drop H2O
                    tt2=data.exper.(useField).param.trial_events.trial{k}(end,3);
                    kk=kk+1;
                else % what is result 2/3?
                    tt2=data.exper.(useField).param.trial_events.trial{k}(:,3);
                    if length(tt2)>1
                        tt2=tt2(1);
                    end
                    kk=kk+1;
                end
            else
                % try to find missing trial_events
                if Result(k)==2 && k<CountedTrial
                    error(['no trial events in odor_rlwm for trial ' num2str(k) ', in file:' filename]);
%                     tt3=data.exper.odor_rlwm.param.trial_events.trial{k+1}(1,3);
%                     temp_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<tt3, 2:4);
%                     False =[36 5;37 5;38 5;39 3;40 3;41 3];
%                     for j=1:size(temp_te,1)
%                         if sum(prod(double(repmat(temp_te(j,2:3),size(False,1),1)==False),2))
%                             tt2=temp_te(j,1);
%                             data.exper.odor_rlwm.param.trial_events.trial{k}=temp_te(j,[2 3 1]);
%                             exper=data.exper;
%                             save([filename '_New.mat'],'exper');
%                             disp(['Saved a new version of ' filename ' to ' filename '_New.mat' ]);
%                         end
%                     end
                elseif Result(k)==0 && k<CountedTrial
                % Result(k) is 0, check if trial number jumped by looking
                % at number of trial transition to next outcome
                    tt3=data.exper.(useField).param.trial_events.trial{k+1}(1,3);
                    temp_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<tt3, 2:4);
                    trial_transition =[512 8];
                    trial_transition_count=0;
                    for j=1:size(temp_te,1)
                        if sum(prod(double(repmat(temp_te(j,2:3),size(trial_transition,1),1)==trial_transition),2))
                            trial_transition_count=trial_transition_count+1;
                            if trial_transition_count==2
                                tt2=temp_te(j,1);
                            end
                        end
                    end
                    if trial_transition_count==1
                        % trial number jumped, only one trial transition to
                        % next outcome, ignore current trial k
                        
                    elseif trial_transition_count==2
                        % missing result from current trial, check if odor
                        % is delivered and find out the outcome. But just
                        % error message for now
                        error(['no trial events in odor_rlwm for trial ' num2str(k) ', in file:' filename]);
                    end
                else
                    error(['no trial events in odor_rlwm for trial ' num2str(k) ', in file:' filename]);
                end
                % check if the Result(k) is 0, if not, manually add back
                % trial events that corresponding to outcome
            end
        end
        % trial_events = (trial, time, state, chan, next state))
        current_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<=tt2, 2:4);
        % C1in in ITI
        c1in_time=current_te(ismember(current_te(:,2),[9 19 512 0 1 11])&ismember(current_te(:,3),1),1);
        % OdorOn Time
        if data.exper.(useField).param.delayodor.value==1
            new_trial_OdorOn_time=current_te(ismember(current_te(:,2),[2 12 22])&ismember(current_te(:,3),8),1);
        else
            new_trial_OdorOn_time=current_te(ismember(current_te(:,2),[1 11 21])&ismember(current_te(:,3),8),1);
            error(['Warning: "delay odor" setting in file:' filename 'was disabled']);
        end
        if ~isempty(new_trial_OdorOn_time)
            if length(new_trial_OdorOn_time)==1
            elseif length(new_trial_OdorOn_time)==2 %&& Result(k-1)~=0
                %  no trial transition ([512 8]) but odor on twice, need to
                %  assume the odor identity based on L/R trial if schedule
                %  has only two odors around current trial
                new_trial_OdorOn_time=new_trial_OdorOn_time(2);
            elseif length(new_trial_OdorOn_time)==2 %&& Result(k-1)==0
                %  no trial transition ([512 8]) but odor on twice, check
                %  previous trial odor on and trial_events
                new_trial_OdorOn_time=new_trial_OdorOn_time(2);
            end
            valid_trials(k)=1;
            % ITI_te is events before new_trial_OdorOn_time
            ITI_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<new_trial_OdorOn_time & ismember(trial_events(:,4),1:6), 2:4);
            last_poke_out=find(ismember(ITI_te(:,3),[4 6]),1,'last');
            if ~isempty(last_poke_out)
                ITI_te(last_poke_out,3)=ITI_te(last_poke_out,3)*10+ITI_te(last_poke_out,3);
            end
            RLWM_EventTimes(:,RLWM_EventTimes_n+1:RLWM_EventTimes_n+length(ITI_te(:,1)))=[ITI_te(:,3)';ITI_te(:,1)';ones(size(ITI_te(:,1)'))*(kk-0.5)];
            RLWM_EventTimes_n=RLWM_EventTimes_n+length(ITI_te(:,1));

            % new_trial_OdorOn_time trigger a new trial [event=7.0x , time, trial=k]
            OdorID=OdorChannelSchedule(k)/100;
            RLWM_EventTimes(:,RLWM_EventTimes_n+1)=[7+OdorID ;new_trial_OdorOn_time;kk]; % new trial odor on
            RLWM_EventTimes_n=RLWM_EventTimes_n+1;

            %Done with ITI trial events, now look at trial events in Trial K
            Tk_te=trial_events(trial_events(:,2)>new_trial_OdorOn_time & trial_events(:,2)<=tt2 & ismember(trial_events(:,4),1:6), 2:4);
            Tk_te1=trial_events(trial_events(:,2)>new_trial_OdorOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==45 & trial_events(:,4)==8), 2:4);
            Tk_te1(:,3)=9.01;
            Tk_te2=trial_events(trial_events(:,2)>new_trial_OdorOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==44 & trial_events(:,4)==8), 2:4);
            Tk_te2(:,3)=9.02;
            Tk_te3=trial_events(trial_events(:,2)>new_trial_OdorOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==43 & trial_events(:,4)==8), 2:4);
            Tk_te3(:,3)=9.03;
            Tk_te=[Tk_te;Tk_te1;Tk_te2;Tk_te3];
            [Y I]=sort(Tk_te(:,1),1);
            Tk_te=Tk_te(I,:);
            RLWM_EventTimes(:,RLWM_EventTimes_n+1:RLWM_EventTimes_n+length(Tk_te(:,1)))=[Tk_te(:,3)';Tk_te(:,1)';ones(size(Tk_te(:,1)'))*kk];
            RLWM_EventTimes_n=RLWM_EventTimes_n+length(Tk_te(:,1));

            %Now look at outcome (tt2) if not already added [event=80+result, time, trial=k]
            RLWM_EventTimes(:,RLWM_EventTimes_n+1)=[80+Result(k);tt2;kk]; %
            RLWM_EventTimes_n=RLWM_EventTimes_n+1;
        elseif tt1==tt2
            % skip current trial k
        else
            error(['no odor on time for trial: ' num2str(k) ' in file: ' filename]);
        end

    end
    out.RLWM_EventTimes=RLWM_EventTimes;

    out.odor_name=OdorName(valid_trials);
    out.odor_dur=str2double(StimParam(schedule(valid_trials),6))';
    out.schedule=schedule(valid_trials);
    portside(LeftRewardP==-1 & RightRewardP==-1)=-1;
    out.portside=portside(valid_trials);
    out.result=Result(valid_trials);
    startTime = data.exper.control.param.trialstart.value;
    out.startTime = startTime(4)*3600+startTime(5)*60+startTime(6);
else
    disp('file not found');
end
