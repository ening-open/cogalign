function get_rdm_48object(rsm_dir, rdm_dir, model_tags)
% Convert model RSMs to RDMs for the 48-object subset.
script_dir = fileparts(mfilename('fullpath'));
load(fullfile(script_dir, 'words.mat'))
load(fullfile(script_dir, 'words48.mat'))

[~, ~, wordposition48] = intersect(words48, words, 'stable');

if ~isfolder(rdm_dir)
    [created, message] = mkdir(rdm_dir);
    if ~created
        error('Cannot create RDM output directory: %s', message);
    end
end

for i = 1:numel(model_tags)
    filename = model_tags{i};
    load(fullfile(rsm_dir, ['RSM_', filename, '_48object.mat']));
    load(fullfile(rsm_dir, ['RSM_', filename, '_48object_half.mat']));
    RSM48_triplet = RSM1854_triplet(wordposition48, wordposition48);
    RSM48_triplet_split1 = RSM1854_triplet_split1(wordposition48, wordposition48);
    RSM48_triplet_split2 = RSM1854_triplet_split2(wordposition48, wordposition48);
    RDM48_triplet = 1 - RSM48_triplet;
    RDM48_triplet_split1 = 1 - RSM48_triplet_split1;
    RDM48_triplet_split2 = 1 - RSM48_triplet_split2;
    save_filename1 = fullfile(rdm_dir, ['RDM_', filename, '_48object.mat']);
    save(save_filename1, 'RDM48_triplet', '-v7');
    save_filename2 = fullfile(rdm_dir, ['RDM_', filename, '_48object_half.mat']);
    save(save_filename2, 'RDM48_triplet_split1', 'RDM48_triplet_split2', '-v7');
end
end
