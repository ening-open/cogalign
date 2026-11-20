function get_human_behavior_consistency(model_rdm_dir, human_rdm_dir, model_tags)
% Print the human and model consistency scores and confidence bounds.
script_dir = fileparts(mfilename('fullpath'));
original_path = path;
restore_path = onCleanup(@() path(original_path));
addpath(fullfile(script_dir, 'nmi_utils'))

load(fullfile(human_rdm_dir, 'RDM_human_48object.mat'));
load(fullfile(human_rdm_dir, 'RDM_human_48object_half.mat'));
RDM_human = RDM48_triplet;
RDM_human_half1 = RDM48_triplet_split1;
RDM_human_half2 = RDM48_triplet_split2;

filenames = model_tags(:).';

% Initialize arrays to store results
rho_values = zeros(length(filenames), 1);
ci_lower = zeros(length(filenames), 1);
ci_upper = zeros(length(filenames), 1);

for file_id = 1:length(filenames)
    filename = filenames{file_id};

    load(fullfile(model_rdm_dir, ['RDM_', filename, '_48object', '.mat']));
    load(fullfile(model_rdm_dir, ['RDM_', filename, '_48object_half', '.mat']));

    RDM = RDM48_triplet;
    RDM_half1 = RDM48_triplet_split1;
    RDM_half2 = RDM48_triplet_split2;

    % compute the human consistency  [ref. Rajalingham et al., 2018, JNeurosci]
    r48_mh = corr(squareformq(RDM),squareformq(RDM_human));
    % run 1000 bootstrap samples for confidence intervals, bootstrap cannot
    % be done across objects because otherwise it's biased
    rng(2)
    rnd = randi(nchoosek(48,2),nchoosek(48,2),1000);
    c1 = squareformq(RDM);
    c2 = squareformq(RDM_human);
    for i = 1:1000
        r48_boot(:,i) = corr(c1(rnd(:,i)),c2(rnd(:,i)));
    end
    r48_mh_ci95_lower = tanh(atanh(r48_mh) - 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI
    r48_mh_ci95_upper = tanh(atanh(r48_mh) + 1.96*std(atanh(r48_boot),[],2)); % reflects 95% CI

    r48_mm = corr(squareformq(RDM_half1),squareformq(RDM_half2));
    r48_hh = corr(squareformq(RDM_human_half1),squareformq(RDM_human_half2));

    numerator = r48_mh;
    denominator = sqrt(r48_mm * r48_hh);
    rho = numerator / denominator

    % Store results
    rho_values(file_id) = rho;
    ci_lower(file_id) = r48_mh_ci95_lower + (rho-r48_mh);
    ci_upper(file_id) = r48_mh_ci95_upper + (rho-r48_mh);

    % add human-human consistency
    %%%%%%%%%%%%%%%
    c1_hh = squareformq(RDM_human_half1);
    c2_hh = squareformq(RDM_human_half2);
    for i = 1:1000
        r48_boot_hh(:,i) = corr(c1_hh(rnd(:,i)),c2_hh(rnd(:,i)));
    end
    r48_hh_ci95_lower = tanh(atanh(r48_hh) - 1.96*std(atanh(r48_boot_hh),[],2)); % reflects 95% CI
    r48_hh_ci95_upper = tanh(atanh(r48_hh) + 1.96*std(atanh(r48_boot_hh),[],2)); % reflects 95% CI
    %%%%%%%%%%%%%
end
rho_values = [r48_hh; rho_values];
ci_lower = [r48_hh_ci95_lower; ci_lower];
ci_upper = [r48_hh_ci95_upper; ci_upper];
filenames = [{'human'}, filenames];

for i = 1:length(filenames)
    filename = filenames{i}
    rho = rho_values(i)
    cil = ci_lower(i)
    ciu = ci_upper(i)
    disp('==========')
end
end
