import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data, InMemoryDataset, HeteroData
import numpy as np 
import os
from tqdm import tqdm
import pickle
from enum import Enum
import random
import copy
import config
from copy import deepcopy


import torch.nn.functional as F
import torch_geometric.loader as geom_data


print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

T = 10

class ActionOfInterest(Enum):
    TARGET=1
    TYPE=2
    TIMING=3



class CombinedDataset(InMemoryDataset):
    def __init__(self, datasets):
        super(CombinedDataset, self).__init__()
        self.data_list = self._combine_datasets(datasets)
        self.data, self.slices = self.collate(self.data_list)
        
    def _combine_datasets(self, datasets):
        data_list = []
        for dataset in datasets:
            data_list.extend([data for data in dataset])
        return data_list


     

class SummerschoolDataset(InMemoryDataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, lookback=20, action_of_interest=ActionOfInterest.TARGET, kept_indices=None, name_filter='', use_samples=1, augment_mfcc=False, no_time=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        #super().__init__(root, transform, pre_transform)
        self.test = test
        self.filename = filename
        self.lookback = lookback
        self.use_samples = use_samples
        self.num_glob_features = 2
        self.filter = kept_indices
        self.name_filter = name_filter
        self.action_of_interest = action_of_interest
        self.no_time = no_time
        self.augment_mfcc = augment_mfcc
        print(kept_indices, config.FEATURE_DICT)
        # double check if mfcc features are among the selected indices, if not, we cannot augment
        if not kept_indices[config.FEATURE_DICT['mfcc'][0]]  and not kept_indices[config.FEATURE_DICT['mfcc_Std'][0]]:
            self.augment_mfcc = False
            if augment_mfcc:
                print(kept_indices, config.FEATURE_DICT['mfcc'][0])
                print('Augmentation cannot be in place because MFCC features are not among selected.')
        
        super(SummerschoolDataset, self).__init__(root, transform, pre_transform)
        

        # last line loads the data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        return ['data'+str(self.lookback)+'_'+str(self.no_time)+'_'+self.name_filter+''+self.action_of_interest.name+('_augmented' if self.augment_mfcc else '')+'.pt']

    def download(self):
        pass

    def get_MFCC_consts(self):
        MFCC_CHANNELS = config.FEATURE_DICT['mfcc'][1] - config.FEATURE_DICT['mfcc'][0]
        mfcc_mean_start_index = -1
        mfcc_std_start_index = -1 
        try:
            # count how many items are true before MFCC features start
            mfcc_mean_start_index = np.sum(self.filter[0:config.FEATURE_DICT['mfcc'][0]]) #self.filter.index(config.FEATURE_DICT['mfcc'][0])
        except Exception as e:
            print(e)
            pass
        try:
            mfcc_std_start_index = np.sum(self.filter[0:config.FEATURE_DICT['mfcc_Std'][0]]) #self.filter.index(config.FEATURE_DICT['mfcc_Std'][0])
        except Exception as e:
            print(e)
            pass
        #print(MFCC_CHANNELS, mfcc_mean_start_index, mfcc_std_start_index)
        return MFCC_CHANNELS, mfcc_mean_start_index, mfcc_std_start_index

    def compute_frequency_block_augmentation(self, X):
        f = int(random.uniform(0,5))
        
        
        X_alt = copy.deepcopy(X)
        # if mfcc channels are in first index
        # replace mean
        MFCC_CHANNELS, mfcc_mean_start_index, mfcc_std_start_index = self.get_MFCC_consts()
        #print(mfcc_mean_start_index, self.filter)

        f0 = int(random.uniform(0, MFCC_CHANNELS-f))
        for p in range(X.shape[0]):
            #print('Freq', f0, f)
            if mfcc_mean_start_index != -1:
                X_alt[p, :,mfcc_mean_start_index+f0:mfcc_mean_start_index+f0+f] = 0 # do we know if it is normalized to have mean 0 or is it different?
            # replace std
            if mfcc_std_start_index != -1:
                X_alt[p, :,mfcc_std_start_index+f0:mfcc_std_start_index+f0+f] = 0 # do we know if it is normalized to have mean 0 or is it different?
            #print(X[p,:,:])
            #print(X_alt[p,:,:])
        return X_alt

    def compute_time_block_augmentation(self, X):
        MFCC_CHANNELS, mfcc_mean_start_index, mfcc_std_start_index = self.get_MFCC_consts()
        tau = X.shape[0] # assuming time is on axis 1
        T_eff = min(T, tau)
        t = int(random.uniform(0,T_eff))
        t0 = int(random.uniform(0, tau-t))
        X_alt = copy.deepcopy(X)
        
        for p in range(X.shape[0]):
            #print('Time', t0, t)
            if mfcc_mean_start_index != -1:
                X_alt[p, t0:t0+t,mfcc_mean_start_index:mfcc_mean_start_index+MFCC_CHANNELS] = 0 # do we know if it is normalized to have mean 0 or is it different?
            if mfcc_std_start_index != -1:
                X_alt[p, t0:t0+t,mfcc_std_start_index:mfcc_std_start_index+MFCC_CHANNELS] = 0
            #print(X[p,:,:])
            #print(X_alt[p,:,:])
        return X_alt


    def create_aug_node(self, data_x, look_back=1):
        #print(data_x.shape)
        aug = random.sample([0,1,2],1)[0]
        if aug == 0:
            a_aug = self.compute_frequency_block_augmentation(data_x)
        elif aug == 1:
            a_aug = self.compute_time_block_augmentation(data_x)
        else:
            a_aug = self.compute_time_block_augmentation(self.compute_frequency_block_augmentation(data_x))
        #dataY.append(dataset[i + look_back, 0])
        return np.array(a_aug) #, numpy.array(dataY)

    def activate_transform(self):
        self.transform = self.transform_function

    def transform_function(self, data):
        if self.transform_scaler == None:
            return data
        data.x = torch.Tensor(self.transform_scaler[0].transform(data.x))
        #data.edge_attr = torch.Tensor(self.transform_scaler[1].transform(data.edge_attr))
        data.global_attr = torch.Tensor(self.transform_scaler[1].transform(data.global_attr))
        return data

    def process(self):
        data_raw = pd.read_csv(self.raw_paths[0])
        state_features_per_participant = []
        state_feature_names = [s for s in data_raw.columns if 'state' in s]
        print(state_feature_names, len(state_feature_names))
        feature_per_participant = int((len(state_feature_names)-2)/3)
        
        curr_end = self.num_glob_features+feature_per_participant
        curr_start = self.num_glob_features
        #print(feature_per_participant, curr_start, curr_end)
        for i in range(3):
            state_features_per_participant.append(state_feature_names[curr_start:curr_end])
            curr_start = curr_end
            curr_end += feature_per_participant
        #print(state_features_per_participant)
        b_file = open("../dicts/dict_type.pickle", "rb")
        dict_org = pickle.load(b_file)
        dict_meaning = dict([(value, key) for key, value in dict_org.items()])
        data_list = []
        node_feats_collect = None
        global_attr_collect = None
        set_collect = False
        statistics_label = [0]*4
        
        for index, row in tqdm(data_raw.iterrows(), total=data_raw.shape[0]):
            
            # Get node features
            node_feats = self._get_node_features(row, state_features_per_participant)
            global_attr = row[state_feature_names[:self.num_glob_features]]
            #global_attr= pd.concat([global_attr, pd.Series([row['action_type']])])
            #print(np.array(node_feats).shape)
            if set_collect:
                node_feats = np.array(node_feats, dtype=np.float)
                node_feats = np.expand_dims(node_feats, axis=1)
                node_feats_collect = np.append(node_feats_collect,node_feats, 1)
                global_attr = np.expand_dims(np.array(global_attr,dtype=np.float), axis=0)
                global_attr_collect = np.append(global_attr_collect, global_attr ,0)
                #print(node_feats_collect.shape)
            else:
                node_feats_collect = np.array(node_feats, dtype=np.float)
                node_feats_collect = np.expand_dims(node_feats_collect, axis=1)
                global_attr_collect = np.array(global_attr, dtype=np.float)
                global_attr_collect = np.expand_dims(global_attr_collect, axis=0)
                #print(node_feats_collect.shape)
                set_collect = True
            # get episode ID of datapoint
            episode_origin = self._get_episode_id(row)
            episode_group_name = self._get_episode_group_name(row)
            # Get adjacency info
            edge_index = self._get_adjacency_info(row)

            # if the target is 0 it means no target -> no action
            decision_taken = row["action_target"]>0


            label_who = row["action_target"]-1
            label_what = row['action_type']-1
        
            label_when = row["action_target"] if row["action_target"] == 0 else 1
            #if dict_meaning[row["action_type"]] == 'NONE': 
            #    label = 0#self._get_labels(row["action_type"])
            #else:
            #    label = 1

            # Create data object
            if node_feats_collect.shape[1] < self.lookback:
                continue
            
            if not self.action_of_interest == ActionOfInterest.TIMING and not decision_taken:
                continue
            #print(node_feats_collect.shape)
            #print(global_attr_collect[-self.lookback:, :].shape)
            node_feats_collect = np.nan_to_num(node_feats_collect)
            global_attr_collect = np.nan_to_num(global_attr_collect)
            if np.count_nonzero(np.isnan(node_feats_collect[:,-self.lookback:, :])) > 0:
                print('OH MY GOD!!!!!!!!!!!!!!!!!')
                print(node_feats_collect[:,-self.lookback:, :])
            if np.count_nonzero(np.isnan(global_attr_collect[-self.lookback:, :])) > 0:
                print('OH MY GOD GLOB!!!!!!!!!!!!!!!!!')
            # for subsampling the features but keeping the very last, first compute the offset
            # this computation of the offset ensures that the last element is kept
            offset_start = (self.lookback+self.use_samples-1)%self.use_samples
            # for selecting the last "lookback" timestemps subsampled by the use_samples, 
            # the following indexing should be done: input[-lookback+offset_start::use_samples]
            # we use this when creating the data object
            if self.no_time:
                num_nodes = 3
                #print(global_attr_collect[-self.lookback+offset_start::self.use_samples, :].shape, global_attr_collect.shape)
                data = Data(x=torch.tensor(np.min(np.nan_to_num(node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :]),1), dtype=torch.float), 
                            edge_index=edge_index,
                            global_attr = torch.tensor([np.min(np.nan_to_num(global_attr_collect[-self.lookback+offset_start::self.use_samples, :]),0)], dtype=torch.float),
                            y_who=label_who, #torch.nn.functional.one_hot(torch.tensor(label-1), num_classes=4),
                            y_who_one=label_who,
                            y_who_local=label_who,
                            y_what=label_what,
                            y_timing=label_when,
                            episode=episode_origin,
                            num_nodes_loc = [num_nodes],
                            num_nodes = num_nodes,
                            num_edges_loc = [num_nodes*(num_nodes-1)],
                            num_edges = num_nodes*(num_nodes-1),
                            group_name = episode_group_name
                            )
                data_list.append(data)
            else:
                data = Data(x=torch.tensor(np.nan_to_num(node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :]), dtype=torch.float), 
                            edge_index=edge_index,
                            global_attr = torch.tensor(np.nan_to_num(global_attr_collect[-self.lookback+offset_start::self.use_samples, :]), dtype=torch.float),
                            y_who=label_who, #torch.nn.functional.one_hot(torch.tensor(label-1), num_classes=4),
                            y_who_one=label_who,
                            y_who_local=label_who,
                            y_what=label_what,
                            y_timing=label_when,
                            episode=episode_origin,
                            group_name = episode_group_name
                            )
                data_list.append(data)
                if self.augment_mfcc:
                    data = Data(x=torch.tensor(self.create_aug_node(np.nan_to_num(node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :])), dtype=torch.float), 
                            edge_index=edge_index,
                            global_attr = torch.tensor(np.nan_to_num(global_attr_collect[-self.lookback+offset_start::self.use_samples, :]), dtype=torch.float),
                            y_who=label_who, #torch.nn.functional.one_hot(torch.tensor(label-1), num_classes=4),
                            y_who_one=label_who,
                            y_who_local=label_who,
                            y_what=label_what,
                            y_timing=label_when,
                            episode=episode_origin,
                            group_name = episode_group_name
                            )
                    data_list.append(data)

            #statistics_label[label-1] += 1
            node_feats_collect = node_feats_collect[:,-self.lookback:, :]
            global_attr_collect = global_attr_collect[-self.lookback:, :]
            #print(data.x.shape)
            #print(node_feats_collect.shape)
            
            # if self.test:
            #     torch.save(data, 
            #         os.path.join(self.processed_dir, 
            #                      f'data_test_{index}.pt'))
            # else:
            #     torch.save(data, 
            #         os.path.join(self.processed_dir, 
            #                      f'data_{index}.pt'))
        # frequency_upsample = [statistics_label[3]/statistics_label[i] for i in range(3)]
        # print(frequency_upsample)
        # extra_list = []
        # for data_point in data_list:
        #     if data_point.y != 3:
        #         repeat = int(frequency_upsample[data_point.y])
        #         for i in range(repeat):
        #             extra_list.append(Data(x=data_point.x.detach().clone(), 
        #                                     edge_index=data_point.edge_index.detach().clone(),
        #                                     global_attr=data_point.global_attr.detach().clone(),
        #                                     y=data_point.y))
        # data_list.extend(extra_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_episode_id(self, mol):
        return mol['episode_id']

    def _get_episode_group_name(self, mol):
        return mol['episode_name'].split('_')[4]

    def _get_node_features(self, mol, col_names):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        #print(mol, mol.index)
        states0 = col_names[0]
        states1 = col_names[1]
        states2 = col_names[2]
        if self.filter != None:
            # select indices
            states0 = np.array(states0)[self.filter]
            #print(states0.shape, np.array(states1).shape)
            states1 = np.array(states1)[self.filter]
            states2 = np.array(states2)[self.filter]
            pass
        #print(len(states0))
        all_node_feats = []
        all_node_feats.append(mol[states0].values)
        all_node_feats.append(mol[states1].values)
        all_node_feats.append(mol[states2].values)
        #print(all_node_feats.dtype)
        return all_node_feats

    def _get_adjacency_info(self, mol):
        edge_indices = [[0,1],[1,0],[2,1],[1,2],[0,2],[2,0]]
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    #def len(self):
    #    return self.data.shape[0]

    # def get(self, idx):
    #     """ - Equivalent to __getitem__ in pytorch
    #         - Is not needed for PyG's InMemoryDataset
    #     """
    #     if self.test:
    #         data = torch.load(os.path.join(self.processed_dir, 
    #                              f'data_test_{idx}.pt'))
    #     else:
    #         data = torch.load(os.path.join(self.processed_dir, 
    #                              f'data_{idx}.pt'))   
    #     return data

class SummerschoolDatasetFixedSize(SummerschoolDataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, 
                lookback=20, action_of_interest=ActionOfInterest.TARGET, kept_indices=None, 
                name_filter='', use_samples=1, augment_mfcc=False, aggr_func='min', type='who'):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        #super().__init__(root, transform, pre_transform)
        self.test = test
        self.filename = filename
        self.lookback = lookback
        self.use_samples = use_samples
        self.num_glob_features = 2
        self.filter = kept_indices
        self.name_filter = name_filter
        self.action_of_interest = action_of_interest
        self.augment_mfcc = augment_mfcc
        self.aggr_func = aggr_func
        self.type = type
        # double check if mfcc features are among the selected indices, if not, we cannot augment
        if not kept_indices[config.FEATURE_DICT['mfcc'][0]]  and not kept_indices[config.FEATURE_DICT['mfcc_Std'][0]]:
            self.augment_mfcc = False
            if augment_mfcc:
                print(kept_indices, config.FEATURE_DICT['mfcc'][0])
                print('Augmentation cannot be in place because MFCC features are not among selected.')
        
        super(SummerschoolDatasetFixedSize, self).__init__(root, filename,                 
                                    lookback=lookback, 
                                    kept_indices=kept_indices, 
                                    name_filter=name_filter, 
                                    use_samples=use_samples,
                                    augment_mfcc=augment_mfcc)
        

        # last line loads the data
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data_fixed_'+self.type+'_'+str(self.lookback)+'_'+str(self.no_time)+'_'+self.name_filter+''+self.action_of_interest.name+('_augmented' if self.augment_mfcc else '')+'.pt']


    def process(self):
        data_raw = pd.read_csv(self.raw_paths[0])
        state_features_per_participant = []
        state_feature_names = [s for s in data_raw.columns if 'state' in s]
        print(state_feature_names, len(state_feature_names))
        feature_per_participant = int((len(state_feature_names)-2)/3)
        
        curr_end = self.num_glob_features+feature_per_participant
        curr_start = self.num_glob_features
        #print(feature_per_participant, curr_start, curr_end)
        for i in range(3):
            state_features_per_participant.append(state_feature_names[curr_start:curr_end])
            curr_start = curr_end
            curr_end += feature_per_participant
        #print(state_features_per_participant)
        b_file = open("../dicts/dict_type.pickle", "rb")
        dict_org = pickle.load(b_file)
        dict_meaning = dict([(value, key) for key, value in dict_org.items()])
        data_list = []
        node_feats_collect = None
        global_attr_collect = None
        set_collect = False
        statistics_label = [0]*4
        
        for index, row in tqdm(data_raw.iterrows(), total=data_raw.shape[0]):
            
            # Get node features
            node_feats = self._get_node_features(row, state_features_per_participant)
            global_attr = row[state_feature_names[:self.num_glob_features]]
            #global_attr= pd.concat([global_attr, pd.Series([row['action_type']])])
            #print(np.array(node_feats).shape)
            if set_collect:
                node_feats = np.array(node_feats, dtype=np.float)
                node_feats = np.expand_dims(node_feats, axis=1)
                node_feats_collect = np.append(node_feats_collect,node_feats, 1)
                global_attr = np.expand_dims(np.array(global_attr,dtype=np.float), axis=0)
                global_attr_collect = np.append(global_attr_collect, global_attr ,0)
                #print(node_feats_collect.shape)
            else:
                node_feats_collect = np.array(node_feats, dtype=np.float)
                node_feats_collect = np.expand_dims(node_feats_collect, axis=1)
                global_attr_collect = np.array(global_attr, dtype=np.float)
                global_attr_collect = np.expand_dims(global_attr_collect, axis=0)
                #print(node_feats_collect.shape)
                set_collect = True
            # get episode ID of datapoint
            episode_origin = self._get_episode_id(row)
            episode_group_name = self._get_episode_group_name(row)
            # Get adjacency info
            edge_index = self._get_adjacency_info(row)

            # if the target is 0 it means no target -> no action
            decision_taken = row["action_target"]>0


            label_who = row["action_target"]-1
            label_what = row['action_type']-1
        
            label_when = row["action_target"] if row["action_target"] == 0 else 1
            #if dict_meaning[row["action_type"]] == 'NONE': 
            #    label = 0#self._get_labels(row["action_type"])
            #else:
            #    label = 1

            # Create data object
            if node_feats_collect.shape[1] < self.lookback:
                continue
            
            if not self.action_of_interest == ActionOfInterest.TIMING and not decision_taken:
                continue
            #print(node_feats_collect.shape)
            #print(global_attr_collect[-self.lookback:, :].shape)
            node_feats_collect = np.nan_to_num(node_feats_collect)
            global_attr_collect = np.nan_to_num(global_attr_collect)
            if np.count_nonzero(np.isnan(node_feats_collect[:,-self.lookback:, :])) > 0:
                print('OH MY GOD!!!!!!!!!!!!!!!!!')
                print(node_feats_collect[:,-self.lookback:, :])
            if np.count_nonzero(np.isnan(global_attr_collect[-self.lookback:, :])) > 0:
                print('OH MY GOD GLOB!!!!!!!!!!!!!!!!!')
            # for subsampling the features but keeping the very last, first compute the offset
            # this computation of the offset ensures that the last element is kept
            offset_start = (self.lookback+self.use_samples-1)%self.use_samples
            # for selecting the last "lookback" timestemps subsampled by the use_samples, 
            # the following indexing should be done: input[-lookback+offset_start::use_samples]
            # we use this when creating the data object
            target = np.zeros(4)
            target[label_who] = 1

            data_p_rotating_observations = []
            target_rotate_collect = []
            action_target_rotate_collect = []
            for n in range(3):
                target_rotate = 0 if label_who == n else 1
                # the goal is to have a fixed size of two always
                target_rotate_one_hot = np.zeros(2)
                target_regression = np.zeros(4)
                target_regression[label_who] = 1
                target_rotate_one_hot[target_rotate] = 1
                target_rotate_collect.append(target_rotate_one_hot)
                action_target_rotate_collect.append(target_rotate)
                node_feats_fixed = node_feats_collect[n,-self.lookback+offset_start::self.use_samples, :]
                other_agents = [True]*3
                other_agents[n] = False
                if self.aggr_func == 'mean':
                    mean_others = node_feats_collect[other_agents].mean(axis=0)
                elif self.aggr_func == 'min':
                    #print(node_feats_collect.shape)
                    mean_others = node_feats_collect[other_agents,-self.lookback+offset_start::self.use_samples, :].min(axis=0)
                    #print(mean_others.shape)
                elif self.aggr_func == 'max':
                    mean_others,_ = node_feats_collect[other_agents].max(axis=0)
                #print(node_feats.shape)
                #mean_others = np.zeros((1,13))
                #np.max(node_feats[other_agents], axis=0, out=mean_others)
                #print(mean_others.shape)
                # extend node_feats_fixed with mean_others
                node_feats_fixed = np.concatenate((node_feats_fixed, mean_others), axis=1)
                #print(node_feats_fixed.shape)
                data_p_rotating_observations.append(node_feats_fixed)
            if self.type == 'what':
                # incase of the whole group being addressed
                if label_who == 3:
                    node_feats_fixed = node_feats_collect[n,-self.lookback+offset_start::self.use_samples, :]
                    if self.aggr_func == 'mean':
                        mean_all = node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :].mean(axis=0)
                    elif self.aggr_func == 'min':
                        #print(node_feats_collect.shape)
                        mean_all = node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :].min(axis=0)
                        #print(mean_others.shape)
                    elif self.aggr_func == 'max':
                        mean_all,_ = node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :].max(axis=0)
                    node_data = torch.tensor([np.concatenate((mean_all, mean_all), axis=1)], dtype=torch.float)
                    

                else:
                    node_data = torch.tensor([data_p_rotating_observations[label_who]], dtype=torch.float)
            else:
                node_data = torch.tensor(data_p_rotating_observations, dtype=torch.float)
            data = Data(x=node_data,
            # node_data_stacked = np.nan_to_num(node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :])
            # print(node_data_stacked.shape)
            # stacked = np.hstack(node_data_stacked[0,:,:], node_data_stacked[1,:,:])
            # stacked = np.hstack(stacked, node_data_stacked[2,:,:])
            # print(stacked.shape)
            #np.nan_to_num(node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :])
            #data = Data(x=torch.tensor(stacked, dtype=torch.float), 
                        edge_index=edge_index,
                        global_attr = torch.tensor(np.nan_to_num(global_attr_collect[-self.lookback+offset_start::self.use_samples, :]), dtype=torch.float),
                        y_who=label_who, #torch.nn.functional.one_hot(torch.tensor(label-1), num_classes=4),
                        y_who_one=torch.tensor(target_rotate_collect, dtype=torch.float),
                        y_who_local=torch.tensor(action_target_rotate_collect, dtype=torch.long),
                        y_what=label_what,
                        y_timing=label_when,
                        episode=episode_origin,
                        group_name = episode_group_name
                        )
            data_list.append(data)
            target = np.zeros(4) #num_nodes+1
            target[label_who] = 1
            if self.augment_mfcc:
                data = Data(x=torch.tensor(self.create_aug_node(np.nan_to_num(node_feats_collect[:,-self.lookback+offset_start::self.use_samples, :])), dtype=torch.float), 
                        edge_index=edge_index,
                        global_attr = torch.tensor(np.nan_to_num(global_attr_collect[-self.lookback+offset_start::self.use_samples, :]), dtype=torch.float),
                        y_who=label_who, #torch.nn.functional.one_hot(torch.tensor(label-1), num_classes=4),
                        y_who_one=target,
                        y_what=label_what,
                        y_timing=label_when,
                        episode=episode_origin,
                        group_name = episode_group_name
                        )
                data_list.append(data)

            #statistics_label[label-1] += 1
            node_feats_collect = node_feats_collect[:,-self.lookback:, :]
            global_attr_collect = global_attr_collect[-self.lookback:, :]
            #print(data.x.shape)
            #print(node_feats_collect.shape)
            
            # if self.test:
            #     torch.save(data, 
            #         os.path.join(self.processed_dir, 
            #                      f'data_test_{index}.pt'))
            # else:
            #     torch.save(data, 
            #         os.path.join(self.processed_dir, 
            #                      f'data_{index}.pt'))
        # frequency_upsample = [statistics_label[3]/statistics_label[i] for i in range(3)]
        # print(frequency_upsample)
        # extra_list = []
        # for data_point in data_list:
        #     if data_point.y != 3:
        #         repeat = int(frequency_upsample[data_point.y])
        #         for i in range(repeat):
        #             extra_list.append(Data(x=data_point.x.detach().clone(), 
        #                                     edge_index=data_point.edge_index.detach().clone(),
        #                                     global_attr=data_point.global_attr.detach().clone(),
        #                                     y=data_point.y))
        # data_list.extend(extra_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class BrainstormingDataset(InMemoryDataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, test_data=False, group_sizes_train = '2-3', type='who'):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        #super().__init__(root, transform, pre_transform)
        self.test = test
        self.root = root
        self.type = type
        self.foldername = filename
        #self.use_samples = use_samples
        self.num_glob_features = 2
        self.group_sizes_train = group_sizes_train
        self.test_data = test_data
        self.transform_scaler = None
        super(BrainstormingDataset, self).__init__(root, transform, pre_transform)
        

        # last line loads the data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        if self.test_data:
            return self.foldername
        return 'raw/'+self.foldername

    def activate_transform(self):
        self.transform = self.transform_function

    def transform_function(self, data):
        if self.transform_scaler == None:
            return data
        data.x = torch.Tensor(self.transform_scaler[0].transform(data.x))
        data.edge_attr = torch.Tensor(self.transform_scaler[1].transform(data.edge_attr))
        data.global_attr = torch.Tensor(self.transform_scaler[2].transform(data.global_attr))
        return data

    @property
    def processed_file_names(self):
        return ['brainstorming_data_'+self.type+'_'+self.group_sizes_train+'.pt']

    def download(self):
        pass

    def process(self):
        dataset_temp, frequencies_action_type = self.get_timefree_dataset_flex(self._get_timefree_features_as_graph)

        for key in dataset_temp:
                print(len(dataset_temp[key]), key)
                if key == 'observations' or key == 'next_observations':
                    continue
                dataset_temp[key] = np.array(dataset_temp[key])
                print(dataset_temp[key].shape)
        
        # for upsampling and action subsampling for two step classification, check original dataset extraction code
        dataset = dataset_temp['observations']
        print('Observations: ', len(dataset))
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])
        #return dataset

    def _get_timefree_features_as_graph(self, data_raw, num_nodes, action_type, action_target, episode_org): #, decisions_for_glob):
        """ 
        This will return a matrix / 3d array of the shape
        [Number of Nodes, Node Feature size, Number of timestamp in action]
        """
        
        len_features_nodes = None
        len_features_edges = None
        node_feats = self._get_node_features(data_raw, num_nodes)
        edge_index, edge_attr = self._get_adjacency_info(data_raw, num_nodes)
        global_feats = self._get_global_features(data_raw, num_nodes)

        # target as one_hot
        target = np.zeros(num_nodes)
        target[action_target] = 1

        observation = Data(x=node_feats, 
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                global_attr = global_feats,
                                y_what=action_type,
                                y_who_one=target,
                                y_who=action_target,
                                num_nodes_loc = [num_nodes],
                                num_nodes = num_nodes,
                                num_edges_loc = [num_nodes*(num_nodes-1)],
                                num_edges = num_nodes*(num_nodes-1),
                                episode=str(num_nodes)+"_p_"+str(episode_org)
                                ) 
            
        return observation



    def get_no_participants_from_data(self, data_raw):
        return int(data_raw.iloc[0,2])

    def get_action_and_target_from_data(self, data_raw):
        action_type = int(data_raw.iloc[0,3])
        # action target is currently on 5th index
        # for time free version first try person as action
        action_target = int(data_raw.iloc[0,4])
        return action_type, action_target

    def get_timefree_dataset_flex(self, func_for_state_extraction, p_restr=False, no_p_exact=2, upsample_what=False):
        dataset = {'observations': [], 'actions_type': [], 'actions_target': []}
        
        folders_in_folder = [x.path for x in os.scandir(self.root+'/'+self.raw_file_names) if x.is_dir()]
        print(folders_in_folder)
        count_terminals = 0
        actions_type = [0]*6
        group_id = -1
        for folder in folders_in_folder:
            print(folder)
            # group_id is used as the episode id for train,val,test split
            group_id += 1
            files_in_folder = os.listdir(folder)
            # find all files that have state and action
            file_in_folder = [x for x in files_in_folder if x.find('state_action_pairs') > -1 and x.find('norm') == -1]
            if len(file_in_folder) > 0:
                file_in_folder = file_in_folder[0]
            else:
                print('NOT FOUND', folder)
                continue
            first_decision = True
            print(folder, file_in_folder)
            if not os.path.exists(folder+'/'+file_in_folder):
                print('Not found ', folder, file_in_folder)
                continue
            data_raw = pd.read_csv(folder+'/'+file_in_folder, header=None)
            #print(data_raw.head())
            #print(data_raw.iloc[-1,:])
            num_episodes = int(data_raw.iloc[-1,0])+1
            #print(data_raw.iloc[-1,1])
            num_decisions = int(data_raw.iloc[-1,1])+1
            for episode_num in range(num_episodes):
                #print('Aloha', data_raw.iloc[:,0])
                df_episode = data_raw[data_raw.iloc[:,0].values==episode_num]
                if df_episode.shape[0] == 0:
                    print('OH NO, nothing in episode, okay for what', episode_num)
                    continue
                num_decisions = int(df_episode.iloc[-1,1])+1
                #print(df_episode)
                for decision_num in range(num_decisions):
                    df_decision = df_episode[df_episode.iloc[:,1].values==decision_num]
                    if df_decision.shape[0] == 0:
                        print('OH NO, nothin in decision, okay for what', decision_num)
                        #print(df_episode.iloc[:,0:5])
                        continue
                    #print('Decision\n', df_decision)
                    # Get node features for number of nodes (in second index)
                    no_p = self.get_no_participants_from_data(df_decision)
                    #print(no_p)
                    if not p_restr or no_p == no_p_exact:
                        action_type, action_target = self.get_action_and_target_from_data(df_decision)
                        # set binary action gaze vs no gaze
                        # if binary_type:
                        #     action_type = action_type if action_type == 1 else 2
                        actions_type[action_type] += 1
                        #if target == 3:
                        #    print(data_raw, folder, episode)
                        terminal = (decision_num == num_decisions-1)
                        # ensure that action_type starts with index 0, if we have a binary type, we just assigned 1 and 2 (could be 0 and 1 directly but well :D)
                        # binary type means actions: [gaze, no gaze] where 'no gaze' means some form of speech
                        # if we do a two step decision making the second decision does only contain the speech elements which start with and ID of 2
                        # take out binary and change of action_type for now
                        # if binary_type:
                        #     action_type = action_type - 1
                        # else:
                        #     action_type = action_type - 2
                        #action_type = action_type-1
                        # func_for_state_extraction is given on calling this function and returns the state in the correct format
                        state = func_for_state_extraction(df_decision, no_p, action_type=action_type, action_target=action_target, episode_org=folder.split('/')[-1] )
                        # the construction of the dataset is just pro forma, actually all information is in the state and is extracted from the state
                        dataset['observations'].append(state)
                        # Get adjacency info
                        #edge_index = self._get_adjacency_info(data_raw, no_p)
                        # Get labels info
                        # action type is currently on 4th index
                        
                        dataset['actions_target'].append(action_target)
                        dataset['actions_type'].append(action_type)
                       

        print(set(dataset['actions_target']))
        print('Frequency of actions', actions_type)
        print('The dataset has ', count_terminals, 'episodes.')
        return dataset, actions_type

    def _get_adjacency_info(self, data_raw, no_p):
        all_edge_feats = []
        all_edge_indices = []
        num_edge_features = None
        for i in range(no_p):
            for j in range(i+1, no_p):
                e_identifier_forward = ['E'+str(i)+'->'+str(j), [i,j]]
                e_identifier_backward = ['E'+str(j)+'->'+str(i), [j,i]]
                if num_edge_features == None:
                    starting_index_edge_f = int(np.where(data_raw.iloc[0,:].values==e_identifier_forward[0])[0])+1
                    next_index = int(np.where(data_raw.iloc[0,:].values==e_identifier_backward[0])[0])
                    num_edge_features = next_index - starting_index_edge_f
                
                for identifier in [e_identifier_forward, e_identifier_backward]:
                    starting_index_edge = int(np.where(data_raw.iloc[0,:].values==identifier[0])[0])+1
                    all_edge_feats.append(data_raw.iloc[-1, starting_index_edge:starting_index_edge+num_edge_features].values)
                    all_edge_indices.append(identifier[1])
        #print(all_edge_feats)
        #print(all_edge_indices)
        all_edge_feats = np.asarray(all_edge_feats, dtype=np.float64)
        #return torch.tensor(all_edge_indices, dtype=torch.int), torch.tensor(all_edge_feats, dtype=torch.float)
        return all_edge_indices, torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_node_features(self, data_raw, num_nodes):
        """ 
        This will return a matrix / 3d array of the shape
        [Number of Nodes, Number of timestamp in action, Node Feature size]
        """

        len_features = None
        all_node_feats = []
        for i in range(num_nodes):
            p_identifier = 'P'+str(i)
            # take first row and find the identifier 'P0', actual index is on [0][0], 
            # +1 since first feature starts after identifier
            starting_index = int(np.where(data_raw.iloc[0,:].values==p_identifier)[0])+1
            if len_features == None:
                # we always have at least two participants so we can use the identifier of the nxt participant to find the length of the feature vector
                next_index = int(np.where(data_raw.iloc[0,:].values=='P'+str(i+1))[0])
                len_features = next_index - starting_index
            
            all_node_feats.append(data_raw.iloc[-1, starting_index:starting_index+len_features].values)

        return torch.tensor(all_node_feats, dtype=torch.float)
   

    def _get_global_features(self, data_raw, num_nodes):
        """ 
        This will return a matrix / 3d array of the shape
        [Number of Nodes, Number of timestamp in action, Node Feature size]
        """
        
        #for index, row in tqdm(data_raw.iterrows(), total=data_raw.shape[0]):
        
        # take first row and find the identifier 'P0', actual index is on [0][0], 
        # +1 since first feature starts after identifier
        starting_index = int(np.where(data_raw.iloc[0,:].values=='GGA')[0])+1

        # we always have at least two participants so we can use the identifier of the nxt participant to find the length of the feature vector
        #next_index = int(np.where(data_raw.iloc[0,:].values=='P'+str(i+1))[0])
        
        all_glob_feats = data_raw.iloc[-1, starting_index:].values.astype(float)
        #print(all_glob_feats)
        #print(all_node_feats.dtype)
        return torch.tensor(all_glob_feats, dtype=torch.float).unsqueeze(1).T


class BrainstormingHeteroDataset(BrainstormingDataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, action_of_interest=ActionOfInterest.TARGET, test_data=False, group_sizes_train = '2-3'):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        #super().__init__(root, transform, pre_transform)
        self.test = test
        self.root = root
        self.foldername = filename
        #self.use_samples = use_samples
        self.num_glob_features = 2
        self.num_type = 5
        self.action_of_interest = action_of_interest
        self.group_sizes_train = group_sizes_train
        self.binary_type=True
        self.test_data = test_data
        self.transform_scaler = None
        super(BrainstormingHeteroDataset, self).__init__(root, transform, pre_transform) 

    @property
    def processed_file_names(self):
        return ['brainstorming_data'+self.group_sizes_train+'_hetero.pt']


    def process(self):
        dataset_temp, frequencies_action_type = self.get_timefree_dataset_flex(self._get_timefree_features_as_graph_hetero, binary_type=self.binary_type)

        for key in dataset_temp:
                print(len(dataset_temp[key]), key)
                if key == 'observations' or key == 'next_observations':
                    continue
                dataset_temp[key] = np.array(dataset_temp[key])
                print(dataset_temp[key].shape)
        
        # for upsampling and action subsampling for two step classification, check original dataset extraction code
        dataset = dataset_temp['observations']
        print('Observations: ', len(dataset))
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])
        #return dataset

    def _get_timefree_features_as_graph_hetero(self, data_raw, num_nodes, action_type, action_target, episode_org): #, decisions_for_glob):
        """ 
        This will return a matrix / 3d array of the shape
        [Number of Nodes, Node Feature size, Number of timestamp in action]
        """
        
        len_features_nodes = None
        len_features_edges = None
        node_feats = self._get_node_features(data_raw, num_nodes)
        edge_index, edge_attr = self._get_adjacency_info(data_raw, num_nodes)
        global_feats = self._get_global_features(data_raw, num_nodes)

        # target as one_hot
        target = np.zeros(num_nodes)
        target[action_target] = 1
        type_one_hot = np.zeros(num_nodes, self.num_type)
        type_one_hot[action_target, action_type] = 1
        edge_index_one_dir_from_robot = []
        for n in num_nodes:
            edge_index_one_dir_from_robot.append([0,n])

        observation = HeteroData()
        observation['p'].x=node_feats
        observation['r'].x=[]
        observation['p','influences','p'].edge_index=edge_index
        observation['p','influences','p'].edge_attr=edge_attr
        observation['r','affects','p'].edge_index=edge_index_one_dir_from_robot
        observation['r','affects','p'].edge_attr=type_one_hot
        observation['p'].num_nodes_loc = [num_nodes]
        observation['p'].num_nodes = num_nodes
        observation['r'].num_nodes_loc = [1]
        observation['r'].num_nodes = 1
        observation['p','influences','p'].num_edges_loc = [num_nodes*(num_nodes-1)]
        observation['p','influences','p'].num_edges = num_nodes*(num_nodes-1)
        observation['r','affects','p'].num_edges = num_nodes
        observation['r','affects','p'].num_edges_loc = [num_nodes]
        observation.episode=str(num_nodes)+"_p_"+str(episode_org)
        observation.global_attr = global_feats
        
            
        return observation
# fixed size here means fixed size of feature vector not dataset
class BrainstormingDatasetFixedSize(InMemoryDataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, test_data=False, group_sizes_train = '2-3', aggr_func='min', type='who'):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        #super().__init__(root, transform, pre_transform)
        self.test = test
        self.root = root
        self.type = type
        self.foldername = filename
        #self.use_samples = use_samples
        self.num_glob_features = 2
        self.group_sizes_train = group_sizes_train
        self.test_data = test_data
        self.transform_scaler = None
        self.aggr_func = aggr_func
        super(BrainstormingDatasetFixedSize, self).__init__(root, transform, pre_transform)
        

        # last line loads the data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        if self.test_data:
            return self.foldername
        return 'raw/'+self.foldername

    def activate_transform(self):
        self.transform = self.transform_function

    def transform_function(self, data):
        if self.transform_scaler == None:
            return data
        data.x = torch.Tensor(self.transform_scaler.transform(data.x))
        return data

    @property
    def processed_file_names(self):
        return ['brainstorming_data_'+self.type+'_'+self.group_sizes_train+'_'+self.aggr_func+'_fixed.pt']

    def download(self):
        pass

    def process(self):
        dataset_temp, frequencies_action_type = self.get_timefree_dataset_flex(self._get_timefree_features_as_fixed_size)

        for key in dataset_temp:
                print(len(dataset_temp[key]), key)
                if key == 'observations' or key == 'next_observations':
                    continue
                dataset_temp[key] = np.array(dataset_temp[key])
                print(dataset_temp[key].shape)
        
        # for upsampling and action subsampling for two step classification, check original dataset extraction code
        dataset = dataset_temp['observations']
        print('Observations: ', len(dataset))
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])
        #return dataset

    def _get_timefree_features_as_fixed_size(self, data_raw, num_nodes, action_type, action_target, episode_org): #, decisions_for_glob):
        """ 
        This will return a matrix / 3d array of the shape
        [Number of Nodes, Node Feature size, Number of timestamp in action]
        """
        
        len_features_nodes = None
        len_features_edges = None
        node_feats = self._get_node_features(data_raw, num_nodes)
        #edge_index, edge_attr = self._get_adjacency_info(data_raw, num_nodes)
        #global_feats = self._get_global_features(data_raw, num_nodes)

        # target as one_hot
        target = np.zeros(num_nodes)
        target[action_target] = 1

        data_p_rotating_observations = []
        target_rotate_collect = []
        action_target_rotate_collect = []
        for n in range(num_nodes):
            target_rotate = 0 if action_target == n else 1
            # the goal is to have a fixed size of two always
            target_rotate_one_hot = np.zeros(2)
            target_regression = np.zeros(num_nodes)
            target_regression[action_target] = 1
            target_rotate_one_hot[target_rotate] = 1
            target_rotate_collect.append(target_rotate_one_hot)
            action_target_rotate_collect.append(target_rotate)
            node_feats_fixed = node_feats[n]
            other_agents = [True]*num_nodes
            other_agents[n] = False
            if self.aggr_func == 'mean':
                mean_others = node_feats[other_agents].mean(axis=0)
            elif self.aggr_func == 'min':
                mean_others,_ = node_feats[other_agents].min(axis=0)
            elif self.aggr_func == 'max':
                mean_others,_ = node_feats[other_agents].max(axis=0)
            #print(node_feats.shape)
            #mean_others = np.zeros((1,13))
            #np.max(node_feats[other_agents], axis=0, out=mean_others)
            #print(mean_others.shape)
            # extend node_feats_fixed with mean_others
            node_feats_fixed = np.concatenate((node_feats_fixed, mean_others))
            data_p_rotating_observations.append(node_feats_fixed)
        if self.type == 'what':
            node_data = torch.tensor([data_p_rotating_observations[action_target]], dtype=torch.float)
        else:
            node_data = torch.tensor(data_p_rotating_observations, dtype=torch.float)
        observation = Data(x=node_data,
                            y_what=action_type,
                            y_who_one=torch.tensor(target_rotate_collect, dtype=torch.float),
                            y_who_local=torch.tensor(action_target_rotate_collect, dtype=torch.long),
                            y_who = action_target,
                            num_nodes = num_nodes,
                            num_nodes_loc = [num_nodes],
                            episode=str(num_nodes)+"_p_"+str(episode_org)
                            ) 
            # node_feats_fixed = torch.tensor([np.concatenate((node_feats_fixed, mean_others))], dtype=torch.float)
        #     observation = Data(x=node_feats_fixed, 
        #                             y_what=action_type,
        #                             y_who_one=target_rotate_one_hot,
        #                             y_who=target_rotate,
        #                             num_nodes = 1,
        #                             episode=str(num_nodes)+"_p_"+str(episode_org)
        #                             ) 
        #     data_p_rotating_observations.append(observation)
            
        return observation



    def get_no_participants_from_data(self, data_raw):
        return int(data_raw.iloc[0,2])

    def get_action_and_target_from_data(self, data_raw):
        action_type = int(data_raw.iloc[0,3])
        # action target is currently on 5th index
        # for time free version first try person as action
        action_target = int(data_raw.iloc[0,4])
        return action_type, action_target

    def get_timefree_dataset_flex(self, func_for_state_extraction, p_restr=False, no_p_exact=2, upsample_what=False):
        dataset = {'observations': [], 'actions_type': [], 'actions_target': []}
        
        folders_in_folder = [x.path for x in os.scandir(self.root+'/'+self.raw_file_names) if x.is_dir()]
        print(folders_in_folder)
        count_terminals = 0
        actions_type = [0]*6
        group_id = -1
        for folder in folders_in_folder:
            # group_id is used as the episode id for train,val,test split
            group_id += 1
            files_in_folder = os.listdir(folder)
            # find all files that have state and action
            file_in_folder = [x for x in files_in_folder if x.find('state_action_pairs') > -1 and x.find('norm') == -1]
            if len(file_in_folder) > 0:
                file_in_folder = file_in_folder[0]
            else:
                print('NOT FOUND', folder)
                continue
            first_decision = True
            print(folder, file_in_folder)
            if not os.path.exists(folder+'/'+file_in_folder):
                print('Not found ', folder, file_in_folder)
                continue
            data_raw = pd.read_csv(folder+'/'+file_in_folder, header=None)
            #print(data_raw.head())
            #print(data_raw.iloc[-1,:])
            num_episodes = int(data_raw.iloc[-1,0])+1
            #print(data_raw.iloc[-1,1])
            num_decisions = int(data_raw.iloc[-1,1])+1
            for episode_num in range(num_episodes):
                #print('Aloha', data_raw.iloc[:,0])
                df_episode = data_raw[data_raw.iloc[:,0].values==episode_num]
                if df_episode.shape[0] == 0:
                    print('OH NO, nothing in episode, okay for what', episode_num)
                    continue
                num_decisions = int(df_episode.iloc[-1,1])+1
                #print(df_episode)
                for decision_num in range(num_decisions):
                    df_decision = df_episode[df_episode.iloc[:,1].values==decision_num]
                    if df_decision.shape[0] == 0:
                        print('OH NO', decision_num)
                        print(df_episode.iloc[:,0:5])
                        continue
                    #print('Decision\n', df_decision)
                    # Get node features for number of nodes (in second index)
                    no_p = self.get_no_participants_from_data(df_decision)
                    #print(no_p)
                    if not p_restr or no_p == no_p_exact:
                        action_type, action_target = self.get_action_and_target_from_data(df_decision)
                        # set binary action gaze vs no gaze
                        # if binary_type:
                        #     action_type = action_type if action_type == 1 else 2
                        actions_type[action_type] += 1
                        #if target == 3:
                        #    print(data_raw, folder, episode)
                        terminal = (decision_num == num_decisions-1)
                        # ensure that action_type starts with index 0, if we have a binary type, we just assigned 1 and 2 (could be 0 and 1 directly but well :D)
                        # binary type means actions: [gaze, no gaze] where 'no gaze' means some form of speech
                        # if we do a two step decision making the second decision does only contain the speech elements which start with and ID of 2
                        # take out binary and change of action_type for now
                        # if binary_type:
                        #     action_type = action_type - 1
                        # else:
                        #     action_type = action_type - 2
                        # fixed in dataset instead
                        #action_type = action_type-1
                        # func_for_state_extraction is given on calling this function and returns the state in the correct format
                        state = func_for_state_extraction(df_decision, no_p, action_type=action_type, action_target=action_target, episode_org=folder.split('/')[-1] )
                        # the construction of the dataset is just pro forma, actually all information is in the state and is extracted from the state
                        dataset['observations'].append(state)
                        # Get adjacency info
                        #edge_index = self._get_adjacency_info(data_raw, no_p)
                        # Get labels info
                        # action type is currently on 4th index
                        
                        #dataset['actions_target'].append(action_target)
                        #dataset['actions_type'].append(action_type)
                       

        #print(set(dataset['actions_target']))
        #print('Frequency of actions', actions_type)
        #print('The dataset has ', count_terminals, 'episodes.')
        return dataset, actions_type

    def _get_adjacency_info(self, data_raw, no_p):
        all_edge_feats = []
        all_edge_indices = []
        num_edge_features = None
        for i in range(no_p):
            for j in range(i+1, no_p):
                e_identifier_forward = ['E'+str(i)+'->'+str(j), [i,j]]
                e_identifier_backward = ['E'+str(j)+'->'+str(i), [j,i]]
                if num_edge_features == None:
                    starting_index_edge_f = int(np.where(data_raw.iloc[0,:].values==e_identifier_forward[0])[0])+1
                    next_index = int(np.where(data_raw.iloc[0,:].values==e_identifier_backward[0])[0])
                    num_edge_features = next_index - starting_index_edge_f
                
                for identifier in [e_identifier_forward, e_identifier_backward]:
                    starting_index_edge = int(np.where(data_raw.iloc[0,:].values==identifier[0])[0])+1
                    all_edge_feats.append(data_raw.iloc[-1, starting_index_edge:starting_index_edge+num_edge_features].values)
                    all_edge_indices.append(identifier[1])
        #print(all_edge_feats)
        #print(all_edge_indices)
        all_edge_feats = np.asarray(all_edge_feats, dtype=np.float64)
        #return torch.tensor(all_edge_indices, dtype=torch.int), torch.tensor(all_edge_feats, dtype=torch.float)
        return all_edge_indices, torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_node_features(self, data_raw, num_nodes):
        """ 
        This will return a matrix / 3d array of the shape
        [Number of Nodes, Number of timestamp in action, Node Feature size]
        """

        len_features = None
        all_node_feats = []
        for i in range(num_nodes):
            p_identifier = 'P'+str(i)
            # take first row and find the identifier 'P0', actual index is on [0][0], 
            # +1 since first feature starts after identifier
            starting_index = int(np.where(data_raw.iloc[0,:].values==p_identifier)[0])+1
            if len_features == None:
                # we always have at least two participants so we can use the identifier of the nxt participant to find the length of the feature vector
                next_index = int(np.where(data_raw.iloc[0,:].values=='P'+str(i+1))[0])
                len_features = next_index - starting_index
            
            all_node_feats.append(data_raw.iloc[-1, starting_index:starting_index+len_features].values)

        return torch.tensor(all_node_feats, dtype=torch.float)
   

    def _get_global_features(self, data_raw, num_nodes):
        """ 
        This will return a matrix / 3d array of the shape
        [Number of Nodes, Number of timestamp in action, Node Feature size]
        """
        
        #for index, row in tqdm(data_raw.iterrows(), total=data_raw.shape[0]):
        
        # take first row and find the identifier 'P0', actual index is on [0][0], 
        # +1 since first feature starts after identifier
        starting_index = int(np.where(data_raw.iloc[0,:].values=='GGA')[0])+1

        # we always have at least two participants so we can use the identifier of the nxt participant to find the length of the feature vector
        #next_index = int(np.where(data_raw.iloc[0,:].values=='P'+str(i+1))[0])
        
        all_glob_feats = data_raw.iloc[-1, starting_index:].values.astype(float)
        #print(all_glob_feats)
        #print(all_node_feats.dtype)
        return torch.tensor(all_glob_feats, dtype=torch.float).unsqueeze(1).T

class BrainstormingDatasetMinimalContext(BrainstormingDatasetFixedSize):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, test_data=False, group_sizes_train = '2-3', type_task='who'):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        #super().__init__(root, transform, pre_transform)
        self.test = test
        self.root = root
        self.foldername = filename
        #self.use_samples = use_samples
        self.num_glob_features = 2
        self.group_sizes_train = group_sizes_train
        self.test_data = test_data
        self.transform_scaler = None
        super(BrainstormingDatasetMinimalContext, self).__init__(root, filename, test, transform, pre_transform, test_data, group_sizes_train, aggr_func, type_task)
        

        # last line loads the data
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return ['brainstorming_data'+self.group_sizes_train+'_'+'_fixed_minimal.pt']
    
    def _get_timefree_features_as_fixed_size(self, data_raw, num_nodes, action_type, action_target, episode_org): #, decisions_for_glob):
        """ 
        This will return a matrix / 3d array of the shape
        [Number of Nodes, Node Feature size, Number of timestamp in action]
        """
        
        len_features_nodes = None
        len_features_edges = None
        node_feats = self._get_node_features(data_raw, num_nodes)
        #edge_index, edge_attr = self._get_adjacency_info(data_raw, num_nodes)
        #global_feats = self._get_global_features(data_raw, num_nodes)

        # target as one_hot
        target = np.zeros(num_nodes)
        target[action_target] = 1

        data_p_rotating_observations = []
        target_rotate_collect = []
        action_target_rotate_collect = []
        for n in range(num_nodes):
            target_rotate = 0 if action_target == n else 1
            # the goal is to have a fixed size of two always
            target_rotate_one_hot = np.zeros(2)
            target_regression = np.zeros(num_nodes)
            target_regression[action_target] = 1
            target_rotate_one_hot[target_rotate] = 1
            target_rotate_collect.append(target_rotate_one_hot)
            action_target_rotate_collect.append(target_rotate)
            node_feats_fixed = node_feats[n]
            other_agents = [True]*num_nodes
            other_agents[n] = False
            #print(node_feats.shape)
            #mean_others = np.zeros((1,13))
            #np.max(node_feats[other_agents], axis=0, out=mean_others)
            #print(mean_others.shape)
            # extend node_feats_fixed with mean_others
            #node_feats_fixed = np.concatenate((node_feats_fixed, mean_others))
            data_p_rotating_observations.append(np.array(node_feats_fixed))
        #print(np.array(data_p_rotating_observations).shape)
        if self.type == 'what':
            node_data = torch.tensor([data_p_rotating_observations[action_target]], dtype=torch.float)
        else:
            node_data = torch.tensor(data_p_rotating_observations, dtype=torch.float)
        observation = Data(x=node_data,
                            y_what=action_type,
                            y_who_one=torch.tensor(target_rotate_collect, dtype=torch.float),
                            y_who_local=torch.tensor(action_target_rotate_collect, dtype=torch.long),
                            y_who = action_target,
                            num_nodes = num_nodes,
                            num_nodes_loc = [num_nodes],
                            episode=str(num_nodes)+"_p_"+str(episode_org)
                            ) 
            # node_feats_fixed = torch.tensor([np.concatenate((node_feats_fixed, mean_others))], dtype=torch.float)
        #     observation = Data(x=node_feats_fixed, 
        #                             y_what=action_type,
        #                             y_who_one=target_rotate_one_hot,
        #                             y_who=target_rotate,
        #                             num_nodes = 1,
        #                             episode=str(num_nodes)+"_p_"+str(episode_org)
        #                             ) 
        #     data_p_rotating_observations.append(observation)
            
        return observation
    
    