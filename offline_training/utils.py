#!/usr/bin/env python
# coding: utf-8
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
import torch_scatter
import torch_geometric.data as geom_data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import os
from MMPN import MMPN, HandleMMPNType, NodeLevelMMPNTimeFree, NodeLevelMMPN, BasicLinearHandler, BasicLinearHandlerType, FixedLengthLinearHandler, FixedLengthLinearHandlerType 
#from MMPN_old import NodeLevelMMPN
import brainstorming_MMPN
import uuid
import pygsheets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
from torch_geometric.data import Dataset, Data, InMemoryDataset
import torch.optim as optim
from sklearn.metrics import classification_report,confusion_matrix
from pygsheets.utils import numericise_all

import absl.flags
from absl import logging
from ml_collections import ConfigDict
import time
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import *
from dataset import CombinedDataset
import torch.serialization
from torch.nn.modules.loss import MSELoss
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([MSELoss])
else:
    print("Warning: PyTorch 1.9.0 does not support add_safe_globals. Skipping...")


CHECKPOINT_PATH = "/usr/local/checkpoints"

####################### from standford rl library
def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs

def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output

def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output

############################################

def createFeatureBoolDict(selected_features, feature_sel_dict, feature_dict):
    # reset dict
    for key in feature_sel_dict:
        feature_sel_dict[key] = False


    selected_features = sorted(selected_features)
    for feat in selected_features:
        feature_sel_dict[feat] = True
    kept_indices_manual_selection = [False]*37
    for key in feature_sel_dict:
        if feature_sel_dict[key]:
            kept_indices_manual_selection[feature_dict[key][0]:feature_dict[key][1]] = [True]*(feature_dict[key][1]-feature_dict[key][0])
    print(kept_indices_manual_selection)
    return feature_sel_dict, kept_indices_manual_selection


def generateTrainDoubleTestValidDataset(dataset, seed, test_size, valid_size, fixed_group=None):
    episodes = np.unique(dataset.data.episode)
    torch.manual_seed(seed)
    random.seed(seed) 
    # first hold out a fixed percent of sessions from each group
    if fixed_group == None:
        ep_walle = np.unique([x.episode for x in dataset if x.group_name=='WALLE'])
        ep_icub = np.unique([x.episode for x in dataset if x.group_name=='ICUB'])
        ep_r2d2 = np.unique([x.episode for x in dataset if x.group_name=='R2D2'])
        datasets = [ep_walle, ep_icub, ep_r2d2]
    else:
        ep_group = np.unique([x.episode for x in dataset if x.group_name==fixed_group]) 
        datasets = [ep_group]
    
    data_episodes_train = None
    data_episodes_valid = None
    data_episodes_test = None
    data_episodes_test_gen = []
    # percentage of each groups data
    for dataset_local in datasets:
        train_ep, test_ep, _, _ = train_test_split(
                dataset_local,
                [0]*len(dataset_local),
                test_size=test_size,
                random_state=seed
            )
        train_indices = [i for i, x in enumerate(dataset.data.episode) if int(x)  in train_ep]
        
        train_dataset = dataset[train_indices].copy()
        valid_indices, test_indices, train_indices = generateTrainTestValidDatasetMixingAllIndices(train_dataset, seed, test_size, valid_size, fixed_group_size_training=None)
        valid_dataset = train_dataset[valid_indices]
        if data_episodes_valid == None:
            data_episodes_valid = valid_dataset
        else:
            data_episodes_valid = CombinedDataset([data_episodes_valid, valid_dataset])
        test_dataset = train_dataset[test_indices]
        if data_episodes_test == None:
            data_episodes_test = test_dataset
        else:
            data_episodes_test = CombinedDataset([data_episodes_test, test_dataset])
        train_dataset = train_dataset[train_indices]
        if data_episodes_train == None:
            data_episodes_train = train_dataset
        else:
            data_episodes_train = CombinedDataset([data_episodes_train,train_dataset])
        data_episodes_test_gen.extend(test_ep)
        print(test_ep)
    
    print(data_episodes_test_gen)
    test_indices = [i for i, x in enumerate(dataset.data.episode) if int(x) in data_episodes_test_gen]
    test_dataset = dataset[test_indices].copy()
    return data_episodes_train, data_episodes_test, data_episodes_valid, test_dataset
    

# mixing_episodes is the parameter that determines if we split train test valid data by episodes or in the whole dataset
def generateTrainTestValidDataset(dataset, seed, test_size, valid_size, mixing_episodes=True, fixed_group_size_training=None, dataset_size_train_val=None,dataset_size_test=None, folds=False, num_folds=6):
    if mixing_episodes:
        return generateTrainTestValidDatasetMixingAll(dataset, seed, test_size, valid_size, fixed_group_size_training)
    else:
        if folds:
            return generateTrainTestValidEpisodeBasedCrossVal(dataset, seed, test_size, valid_size, fixed_group_size_training,  dataset_size_train_val, dataset_size_test, num_folds=num_folds)
        return generateTrainTestValidEpisodeBased(dataset, seed, test_size, valid_size, fixed_group_size_training,  dataset_size_train_val, dataset_size_test)

def split_episodes_specific_groups_fixed_sizes(episodes, seed, test_size, valid_size, fixed_group_size_training,  dataset_size_train_val, dataset_size_test):
    fixed_group_size_training = fixed_group_size_training.replace('\r','')
    if dataset_size_train_val == None:
        test_ep = [ep for ep in episodes if not ep.startswith(str(fixed_group_size_training)) ]
        train_val_ep = [ep for ep in episodes if ep.startswith(str(fixed_group_size_training)) ]
        train_ep, valid_ep, _, _ = train_test_split(
            train_val_ep,
            [0]*len(train_val_ep),
            test_size=valid_size,
            random_state=seed
        )
    else:
        random.shuffle(episodes)
        random.shuffle(episodes)
        
        if '-' in fixed_group_size_training:
            ep_2 = [ep for ep in episodes if ep.startswith(str(2)) ]
            ep_3 = [ep for ep in episodes if ep.startswith(str(3)) ]
            
            dataset_size_per_group_size = int(dataset_size_train_val/2)
            # in this case we use groups of size 2 and 3 for training (not flexible for groups of other sizes)
            # first separate the test set, the sizes are given in absolute number
            # same number of sessions for 2 and 3
            train_val_2, test_ep_2, _, _ = train_test_split(
                ep_2,
                [0]*len(ep_2),
                test_size=(dataset_size_test/2)/len(ep_2),
                random_state=seed
            )
            train_val_2 = train_val_2[0:dataset_size_per_group_size]
            train_ep_2, valid_ep_2, _, _ = train_test_split(
                train_val_2,
                [0]*len(train_val_2),
                test_size=valid_size,
                random_state=seed
            )

            train_val_3, test_ep_3, _, _ = train_test_split(
                ep_3,
                [0]*len(ep_3),
                test_size=(dataset_size_test/2)/len(ep_3),
                random_state=seed
            )
            train_val_3 = train_val_3[0:dataset_size_per_group_size]
            train_ep_3, valid_ep_3, _, _ = train_test_split(
                train_val_3,
                [0]*len(train_val_3),
                test_size=valid_size,
                random_state=seed
            )

            test_ep = test_ep_2
            test_ep.extend(test_ep_3)
            train_ep = train_ep_2
            train_ep.extend(train_ep_3)
            valid_ep = valid_ep_2
            valid_ep.extend(valid_ep_3)
        else:            
            test_ep = [ep for ep in episodes if not ep.startswith(str(fixed_group_size_training)) ]
            test_ep = test_ep[0:dataset_size_test]
           
            train_val_ep = [ep for ep in episodes if ep.startswith(str(fixed_group_size_training)) ]

            #print(train_val_ep)
            train_val_ep = train_val_ep[0:dataset_size_train_val]
            print('Validation size', valid_size, len(train_val_ep))
            train_ep, valid_ep, _, _ = train_test_split(
                train_val_ep,
                [0]*len(train_val_ep),
                test_size=valid_size,
                random_state=seed
            )

    return train_ep, valid_ep, test_ep

def generateTrainTestValidEpisodeBased(dataset, seed, test_size, valid_size, fixed_group_size_training,  dataset_size_train_val, dataset_size_test):
    episodes = np.unique(dataset.data.episode)
    torch.manual_seed(seed)
    random.seed(seed)
    if fixed_group_size_training == None:
        train_ep, test_ep, _, _ = train_test_split(
            episodes,
            [0]*len(episodes),
            test_size=test_size,
            random_state=seed
        )
        train_ep, valid_ep, _, _ = train_test_split(
            train_ep,
            [0]*len(train_ep),
            test_size=valid_size,
            random_state=seed
        )
    else:
        train_ep, valid_ep, test_ep = split_episodes_specific_groups_fixed_sizes(episodes, seed, test_size, valid_size, fixed_group_size_training, dataset_size_train_val, dataset_size_test)

    if not type(dataset.data.episode[0]) == torch.Tensor:
        test_indices = [i for i, x in enumerate(dataset.data.episode) if x in test_ep]
        valid_indices = [i for i, x in enumerate(dataset.data.episode) if x in valid_ep]
        train_indices = [i for i, x in enumerate(dataset.data.episode) if x  in train_ep]
    else:
        test_indices = [i for i, x in enumerate(dataset.data.episode) if int(x) in test_ep]
        valid_indices = [i for i, x in enumerate(dataset.data.episode) if int(x) in valid_ep]
        train_indices = [i for i, x in enumerate(dataset.data.episode) if int(x) in train_ep]
    valid_dataset = dataset[valid_indices].copy()
    test_dataset = dataset[test_indices].copy()
    train_dataset = dataset[train_indices].copy()
    print('Train: ', np.unique(train_dataset.data.episode))
    print('Valid: ', np.unique(valid_dataset.data.episode))
    print('Test: ', np.unique(test_dataset.data.episode)) #, valid_indices, train_indices)
    
    return train_dataset, test_dataset, valid_dataset, 'episode_based'

def generateTrainTestValidEpisodeBasedCrossVal(dataset, seed, test_size, valid_size, fixed_group_size_training,  dataset_size_train_val, dataset_size_test, num_folds=6):
    episodes = np.unique(dataset.data.episode)
    torch.manual_seed(seed)
    random.seed(seed)
    if fixed_group_size_training == None:
        train_ep_temp, test_ep, _, _ = train_test_split(
            episodes,
            [0]*len(episodes),
            test_size=test_size,
            random_state=seed
        )
        train_ep_temp, valid_ep_temp, _, _ = train_test_split(
            train_ep_temp,
            [0]*len(train_ep_temp),
            test_size=valid_size,
            random_state=seed
        )
    else:
        train_ep_temp, valid_ep_temp, test_ep = split_episodes_specific_groups_fixed_sizes(episodes, seed, test_size, valid_size, fixed_group_size_training, dataset_size_train_val, dataset_size_test)
    # construct folds, we throw train and valid ep together to ensure that the same data is used as when not using cross validation
    train_valid_ep = train_ep_temp
    train_valid_ep.extend(valid_ep_temp)
    train_valid_ep = np.array(train_valid_ep)
    train_valid_datasets = []
    np.random.shuffle(train_valid_ep)
    if '-' in fixed_group_size_training:
        # ensure that one of each group size is in valid vs train
        ep_2 = [ep for ep in train_valid_ep if ep.startswith(str(2)) ]
        ep_3 = [ep for ep in train_valid_ep if ep.startswith(str(3)) ]
        splits_2 = np.array_split(ep_2, num_folds)
        splits_3 = np.array_split(ep_3, num_folds)
        splits = []
        for i in range(num_folds):
            splits.append(np.concatenate([splits_2[i], splits_3[i]]))
        splits = np.array(splits)
    else:
        splits = np.array_split(train_valid_ep, num_folds)
    
    
    for n in range(num_folds):
        valid_ep = splits[n]
        train_ep = np.concatenate([splits[i] for i in range(num_folds) if i != n])
        valid_indices = [i for i, x in enumerate(dataset.data.episode) if x in valid_ep]
        train_indices = [i for i, x in enumerate(dataset.data.episode) if x in train_ep]
        
        valid_dataset = dataset[valid_indices].copy()
        
        train_dataset = dataset[train_indices].copy()
        print('For fold ', n)
        print(np.unique(train_dataset.data.episode))
        print(np.unique(valid_dataset.data.episode))
        train_valid_datasets.append([train_dataset, valid_dataset])
    
    test_indices = [i for i, x in enumerate(dataset.data.episode) if x in test_ep]
    test_dataset = dataset[test_indices].copy()
    print(np.unique(test_dataset.data.episode)) 
    
    return train_valid_datasets, test_dataset, 'episode_based'
    
def generateTrainTestValidDatasetMixingAllIndices(dataset, seed, test_size, valid_size, fixed_group_size_training):
    if fixed_group_size_training != None:
        raise NotImplementedError
    dataset.shuffle()
    torch.manual_seed(seed)
    print(dataset)

    print(len(dataset),
       dataset.data.y_who.shape)
    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.data.y_who,
        stratify=dataset.data.y_who,
        test_size=test_size,
        random_state=seed
    )
    print(train_indices)
    train_indices, valid_indices, _, _ = train_test_split(
        train_indices,
        dataset.data.y_who[train_indices],
        stratify=dataset.data.y_who[train_indices],
        test_size=valid_size,
        random_state=seed
    )
    return valid_indices, test_indices, train_indices


def generateTrainTestValidDatasetMixingAll(dataset, seed, test_size, valid_size, fixed_group_size_training):
    valid_indices, test_indices, train_indices = generateTrainTestValidDatasetMixingAllIndices(dataset, seed, test_size, valid_size, fixed_group_size_training)
    valid_dataset = dataset[valid_indices]
    test_dataset = dataset[test_indices]
    train_dataset = dataset[train_indices]
    return train_dataset, test_dataset, valid_dataset, 'random_based'

from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F

def upsampleTrain(train_dataset, num_classes, lambda_y):
    statistics_label = [0] * num_classes
    for data_point in train_dataset:
        statistics_label[lambda_y(data_point)] += 1

    max_class = statistics_label.index(max(statistics_label))
    print("[Class distribution]:", statistics_label)
    frequency_upsample = [
        statistics_label[max_class] / statistics_label[i] if statistics_label[i] > 0 else 1
        for i in range(num_classes)
    ]
    print("[Upsample ratios]:", frequency_upsample)

    extra_list = []
    list_data_points = []

    for data_point in train_dataset:
        # 添加 y_who_one
        if not hasattr(data_point, 'y_who_one'):
            data_point.y_who_one = F.one_hot(data_point.y_who, num_classes=num_classes).float()

        list_data_points.append(data_point)

        if lambda_y(data_point) != max_class:
            repeat = int(frequency_upsample[lambda_y(data_point)])
            for _ in range(repeat):
                # 🛡️ 用 getattr 设置默认值
                new_data = Data(
                    x=data_point.x.detach().clone(),
                    edge_index=data_point.edge_index.detach().clone(),
                    global_attr=data_point.global_attr.detach().clone(),
                    y_who=data_point.y_who,
                    y_who_one=data_point.y_who_one.detach().clone(),
                    y_who_local=getattr(data_point, 'y_who_local', None),
                    y_what=getattr(data_point, 'y_what', None),
                    y_timing=getattr(data_point, 'y_timing', None),
                    episode=getattr(data_point, 'episode', None),
                    group_name=getattr(data_point, 'group_name', None),
                    num_nodes=data_point.num_nodes,
                    num_nodes_loc=[data_point.num_nodes],
                    num_edges=data_point.num_nodes * (data_point.num_nodes - 1),
                    num_edges_loc=[data_point.num_nodes * (data_point.num_nodes - 1)],
                )
                extra_list.append(new_data)

    list_data_points.extend(extra_list)

    # Collate into InMemoryDataset
    dataset = InMemoryDataset("placeholder", None, None)
    dataset.data, dataset.slices = dataset.collate(list_data_points)
    return dataset

def loadData(train_dataset, valid_dataset, test_dataset, h_BATCH_SIZE):
    graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=h_BATCH_SIZE, shuffle=True, num_workers=5)if train_dataset else None
    graph_val_loader = geom_data.DataLoader(valid_dataset, batch_size=h_BATCH_SIZE, num_workers=5)if valid_dataset else None # Additional loader if you want to change to a larger dataset
    graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=h_BATCH_SIZE, num_workers=5)if test_dataset else None
    return graph_train_loader, graph_val_loader, graph_test_loader

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)

def train_graph_classifier_simple(model_name, wandb_project_name, h_SEED, device, graph_train_loader, graph_val_loader, graph_test_loader, early_stopping, tune_lr, max_epochs, load_pretrained=False, pretrained_filename="",**model_kwargs):
    model = brainstorming_MMPN.NodeLevelMMPN(**model_kwargs).to(device)
    optimizer = model.configure_optimizers()
    for batch_idx, batch in enumerate(graph_train_loader):
        loss = model.training_step(batch, batch_idx)
        print(loss)
        loss.backward()
        
        print(model.model.embedding_nn)
        print('Loss', model.model.embedding_nn.weight.grad)
        print('Loss', model.model.update_nn[0].weight.grad)
        print('Loss', model.model.message_nn[0].weight.grad)
        optimizer.step()
        optimizer.zero_grad()


def train_graph_classifier(model_name, wandb_project_name, h_SEED, device, graph_train_loader, graph_val_loader, graph_test_loader, early_stopping, tune_lr, max_epochs, load_pretrained=False, train_pretrained=False, return_path=False, use_scheduler=False, pretrained_filename="",**model_kwargs):
    train = True
    print("Running with model", model_name)
    set_random_seed(h_SEED)
    run_id = uuid.uuid4().hex
    wandb_logger = WandbLogger(project=wandb_project_name, mode="offline", id=run_id)
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    model_checkpoints = os.path.join(root_dir, str(run_id))
    os.makedirs(root_dir, exist_ok=True)
    trainer_callbacks=[
        ModelCheckpoint(
            dirpath=model_checkpoints, 
            save_top_k=10, 
            monitor="val-macro avg-f1-score", 
            mode='max',
            filename='{epoch}-{val_loss:.5f}-{val_acc:.5f}'
            )]
    if early_stopping:
        early_stop_callback = EarlyStopping(monitor="val_acc",  min_delta=-0.07, patience=10, verbose=False, mode="max")
        trainer_callbacks.append(early_stop_callback)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer_callbacks.append(lr_monitor)
    

    
    # Check whether pretrained model exists. If yes, load it and skip training
    #pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
    if load_pretrained: 
        if not train_pretrained:
            train = False
        print("Found pretrained model, loading...")
        pl.seed_everything(h_SEED)
        
        if model_name == "Brainstorming-Linear-Who":
            model = brainstorming_MMPN.BaselineLinear.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "Brainstorming-Linear-What":
            model = brainstorming_MMPN.BaselineLinearType.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "Brainstorming-RandomForest-Who":
            model = brainstorming_MMPN.RandomForestModel.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "NodeLevelMMPN":
            model = NodeLevelMMPN.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "HandleMMPNType":
            model = HandleMMPNType.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "Brainstorming-MMPN-Who":
            model = brainstorming_MMPN.NodeLevelMMPN.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "SingleVectorNetwork":
            model = BasicLinearHandler.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "SingleVectorNetworkType":
            model = BasicLinearHandlerType.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "FixedLengthNetwork":
            model = FixedLengthLinearHandler.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "FixedLengthNetwork-Type":
            model = FixedLengthLinearHandlerType.load_from_checkpoint(pretrained_filename).to(device)
        elif model_name == "Brainstorming-MMPN-What":
            model = brainstorming_MMPN.GlobalLevelMMPN.load_from_checkpoint(pretrained_filename).to(device)
        
    else:
        pl.seed_everything(h_SEED)
        
        if model_name == "Brainstorming-Linear-Who":
            model = brainstorming_MMPN.BaselineLinear(**model_kwargs).to(device)
        elif model_name == "Brainstorming-Linear-What":
            model = brainstorming_MMPN.BaselineLinearType(**model_kwargs).to(device)
        elif model_name == "Brainstorming-RandomForest-Who":
            model = brainstorming_MMPN.RandomForestModel(**model_kwargs).to(device)
        elif model_name == "NodeLevelMMPN":
            model = NodeLevelMMPN(**model_kwargs).to(device)
        elif model_name == "NodeLevelMMPNTimeFree":
            model = NodeLevelMMPNTimeFree(**model_kwargs).to(device)
        elif model_name == "HandleMMPNType":
            model = HandleMMPNType(**model_kwargs).to(device)
        elif model_name == "SingleVectorNetwork":
            model = BasicLinearHandler(**model_kwargs).to(device)
        elif model_name == "SingleVectorNetworkType":
            model = BasicLinearHandlerType(**model_kwargs).to(device)
        elif model_name == "FixedLengthNetwork":
            model = FixedLengthLinearHandler(**model_kwargs).to(device)
        elif model_name == "FixedLengthNetwork-Type":
            model = FixedLengthLinearHandlerType(**model_kwargs).to(device)
        elif model_name == "Brainstorming-MMPN-Who":
            model = brainstorming_MMPN.NodeLevelMMPN(**model_kwargs).to(device)
        elif model_name == "Brainstorming-MMPN-What":
            model = brainstorming_MMPN.GlobalLevelMMPN(**model_kwargs).to(device)
        
    trainer = pl.Trainer(default_root_dir=root_dir,
                        deterministic=True,
                         #callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         callbacks=trainer_callbacks,
                         devices=1 if str(device).startswith("cuda") else "auto",
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         max_epochs=max_epochs, logger=wandb_logger, auto_lr_find=tune_lr)
    if train: 
        print('fitting now')
        
        trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
        trainer.logger.config = model_kwargs
        if train_pretrained:
            test_result = trainer.test(model.to(device), graph_test_loader, verbose=False)
        if tune_lr:
            trainer.tune(model.to(device), graph_train_loader)
       
        trainer.logger.config['lr'] = model.learning_rate
        trainer.fit(model=model, train_dataloaders=graph_train_loader, val_dataloaders=graph_val_loader)
        if model_name == "Brainstorming-Linear-Who":
            model = brainstorming_MMPN.BaselineLinear.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "Brainstorming-Linear-What":
            model = brainstorming_MMPN.BaselineLinearType.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "Brainstorming-RandomForest-Who":
            model = brainstorming_MMPN.RandomForestModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "NodeLevelMMPN":
            model = NodeLevelMMPN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "NodeLevelMMPNTimeFree":
            model = NodeLevelMMPNTimeFree.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "HandleMMPNType":
            model = HandleMMPNType.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "SingleVectorNetwork":
            model = BasicLinearHandler.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "SingleVectorNetworkType":
            model = BasicLinearHandlerType.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "FixedLengthNetwork":
            model = FixedLengthLinearHandler.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "FixedLengthNetworkType":
            model = FixedLengthLinearHandlerType.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "Brainstorming-MMPN-Who":
            model = brainstorming_MMPN.NodeLevelMMPN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)
        elif model_name == "Brainstorming-MMPN-What":
            model = brainstorming_MMPN.GlobalLevelMMPN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(device)

    val_result = trainer.validate(model.to(device), graph_val_loader, verbose=False)
    print("Direct result")
    print(val_result)
    #test_result = trainer.test(model, graph_test_loader, verbose=False)
    test_result = trainer.test(model.to(device), graph_test_loader, verbose=False)
    test_loss = test_result[0].get('test_loss', None)
    
    result = {"val": val_result[0]['val_acc'], "test":  test_result[0]['test_acc']} 
    
    best_ckpt_path = None
    for cb in trainer_callbacks:
        if isinstance(cb, ModelCheckpoint):
            best_ckpt_path = cb.best_model_path
            break

    if return_path:
        return model, result, val_result, test_result, test_loss, run_id, model.num_parameters, best_ckpt_path
    else:
        return model, result, val_result, test_result, test_loss, run_id, model.num_parameters, None



def readWorkSheet(sheet_name, sheet_id):
    gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')
    #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
    sh = gc.open(sheet_name)
    worksheet = sh[sheet_id]
    cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
    # skip first row
    head = cells[0]
    cells = cells[1:]
    
    return pd.DataFrame(cells, columns=head)

def checkIfWritten(sheet_name, sheet_id, run_id):
    print('Checking if written', sheet_name, sheet_id, run_id)
    df_worksheet = readWorkSheet(sheet_name, sheet_id)
    # check if run_id is in the first column of worksheet
    return run_id in df_worksheet[df_worksheet.columns[0]].values

def writeToGoogleSheets(sheet_name, sheet_id, run_id, h_mess_arch_1, 
                        h_node_arch_1, h_mess_arch_2, h_node_arch_2, split_type, 
                        h_lookback, h_SEED, feature_sel_dict, dict_double, dict_double_test, 
                        dataset='Full', use_drive=True):
    
    new_row_info = [run_id, dataset, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, split_type, h_SEED]
    new_row = []
    for key in feature_sel_dict:
        new_row.append(int(feature_sel_dict[key]))
    header_names_val, header_names_test, new_row_val, new_row_test = writeToGoogleSheetsExtractResultsForRow(dict_double, dict_double_test)
    print(dict_double, dict_double_test, new_row_test, new_row_val)

    new_row = new_row_info+new_row+new_row_val+new_row_test
    
    if use_drive:
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')
        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        is_written = False
        while not is_written:
            worksheet.insert_rows(last_row, number=1, values= new_row)
            time.sleep(4)
            if checkIfWritten(sheet_name, sheet_id, run_id):
                is_written = True
    else:
        df = pd.DataFrame([new_row])
        print(df)
    

def writeToGoogleSheetsTradCrossVal(sheet_name, sheet_id, run_id, common_ident, arch, params, 
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc,
                num_decisions, limit_train, pooling_op, use_drive=True):
    
    new_row_info = [run_id, 'Full' if not limit_train else limit_train, num_decisions, pooling_op, \
                    arch, split_type, h_SEED]
    header = ["RunID",	"Training Set", "#Decisions", 'Pooling Operation', \
        "Arch", "SplitType","Seed"] 
    for key in params:
        header.append(key)
        new_row_info.append(params[key])
    new_row_info += [ val_acc, test_acc]
    header += ['Val Accuracy', 'Test Accuracy']
    
    header_names_val, header_names_test, new_row_val, new_row_test = writeToGoogleSheetsExtractResultsForRow(dict_double_val, dict_double_test)
    
    header = header + header_names_val+header_names_test
    new_row = new_row_info+new_row_val+new_row_test
    if use_drive:
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        print(last_row)
        print(header)
        header = ['Common ID']+header
        new_row = [common_ident]+new_row
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        worksheet.insert_rows(last_row, number=1, values= new_row)
    else:
        df = pd.DataFrame([new_row], columns=header)
        print(df)
    

def writeToGoogleSheetsLinearCrossVal(sheet_name, sheet_id, run_id, common_ident, architecture, 
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                num_params, batch_size, loss_module, num_decisions, limit_train, pooling_op, dropout,
                use_drive=True):
    
    new_row_info, header = writeToGoogleSheetsCreateRowInfo(sheet_name, sheet_id, run_id,architecture, 
                        split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                        num_params, batch_size, loss_module, num_decisions, limit_train, pooling_op, dropout)
    
    header_names_val, header_names_test, new_row_val, new_row_test = writeToGoogleSheetsExtractResultsForRow(dict_double_val, dict_double_test)
    
    header = header + header_names_val+header_names_test
    new_row = new_row_info+new_row_val+new_row_test
    gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

    #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
    if use_drive:
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        print(last_row)
        print(header)
        header = ['Common ID']+header
        new_row = [common_ident]+new_row
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        #worksheet.insert_rows(last_row, number=1, values= new_row)
        is_written = False
        while not is_written:
            worksheet.insert_rows(last_row, number=1, values= new_row)
            time.sleep(4)
            if checkIfWritten(sheet_name, sheet_id, common_ident):
                is_written = True
    else:
        df = pd.DataFrame([new_row], columns=header)
        print(df)

def writeToGoogleSheetsCrossValSummaryTradLinear(sheet_name, sheet_id, data_over_folds, architecture, params,
                split_type, h_SEED, limit_train, pooling_op, use_drive=True):
    
    header = ["CommonID",	"Training Set", "#Decisions", "Pooling Operation", \
            "Architecture", "SplitType"	,"Seed"]
    

    num_decision_str = str(np.mean(data_over_folds['num_decisions']))+' +- '+str(np.std(data_over_folds['num_decisions']))
    new_row = [data_over_folds['ident'], 'Full' if not limit_train else limit_train, num_decision_str, pooling_op,
                    architecture, split_type, h_SEED]
    for key in params:
        header.append(key)
        new_row.append(params[key])
    datasets = ['val', 'test']
    values = ['acc', 'macro avg: f1']
    for d in datasets:
        for v in values:
            m = np.mean(data_over_folds[d][v])
            sd = np.std(data_over_folds[d][v])
            header.append('(mean)'+d+': '+v)
            new_row.append(m)
            header.append('(sd)'+d+': '+v)
            new_row.append(sd)
    if use_drive:
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        #print(last_row)
        #print(header)
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        worksheet.insert_rows(last_row, number=1, values= new_row)
    else:
        df = pd.DataFrame([new_row], columns=header)
        print(df)

def readWorkSheet(sheet_name, sheet_id):
    gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

    #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
    sh = gc.open(sheet_name)
    worksheet = sh[sheet_id]
    cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
    # skip first row
    head = cells[0]
    cells = cells[1:]
    #print(head.shape, cells.shape)
    df = pd.DataFrame(cells, columns=head)

    df = df.dropna()
    return df

def checkIfWritten(sheet_name, sheet_id, run_id):
    print('Checking if written', sheet_name, sheet_id, run_id)
    df_worksheet = readWorkSheet(sheet_name, sheet_id)
    # check if run_id is in the first column of worksheet
    print("Now checking")
    print(run_id in df_worksheet[df_worksheet.columns[0]].values)
    return run_id in df_worksheet[df_worksheet.columns[0]].values


def writeToGoogleSheetsTradLinear(sheet_name, sheet_id, run_id, dict_double_val,dict_double_test,val_acc, test_acc, architecture, params,
                split_type, h_SEED, limit_train, pooling_op, use_drive=True):
    
    header = ["RunID",	"Training Set",  "Pooling Operation", \
            "Architecture", "SplitType"	,"Seed"]
    

    new_row = [run_id, 'Full' if not limit_train else limit_train, pooling_op,
                    architecture, split_type, h_SEED]
    for key in params:
        header.append(key)
        new_row.append(params[key])
    new_row += [ val_acc, test_acc]
    header += ['Val Accuracy', 'Test Accuracy']
    header_names_val, header_names_test, new_row_val, new_row_test = writeToGoogleSheetsExtractResultsForRow(dict_double_val, dict_double_test)
    
    header = header + header_names_val+header_names_test
    new_row = new_row+new_row_val+new_row_test
    if use_drive:
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        #print(last_row)
        #print(header)
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        worksheet.insert_rows(last_row, number=1, values= new_row)
    else:
        df = pd.DataFrame([new_row], columns=header)
        print(df)

def writeToGoogleSheetsCrossValSummaryLinear(sheet_name, sheet_id, data_over_folds, architecture,
                split_type, h_SEED, batch_size, loss_module, limit_train, pooling_op, dropout, use_drive=True):
    
    header = ["CommonID",	"Training Set", "#Decisions", "Pooling Operation", \
            "Architecture",	'#Parameters', "SplitType"	,"Seed", "Batch size", \
            "Dropout","Loss func"]
    num_decision_str = str(np.mean(data_over_folds['num_decisions']))+' +- '+str(np.std(data_over_folds['num_decisions']))
    new_row = [data_over_folds['ident'], 'Full' if not limit_train else limit_train, num_decision_str, pooling_op,
                    architecture, data_over_folds['num_params'], split_type, h_SEED, batch_size, \
                    dropout, loss_module]
    datasets = ['val', 'test']
    values = ['acc', 'macro avg: f1']
    for d in datasets:
        for v in values:
            m = np.mean(data_over_folds[d][v])
            sd = np.std(data_over_folds[d][v])
            header.append('(mean)'+d+': '+v)
            new_row.append(m)
            header.append('(sd)'+d+': '+v)
            new_row.append(sd)
    if use_drive:
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        print(last_row)
        print(header)
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        #worksheet.insert_rows(last_row, number=1, values= new_row)
        is_written = False
        while not is_written:
            worksheet.insert_rows(last_row, number=1, values= new_row)
            time.sleep(4)
            if checkIfWritten(sheet_name, sheet_id, data_over_folds['ident']):
                is_written = True
    else:
        df = pd.DataFrame([new_row], columns=header)
        print(df)
#
def writeToGoogleSheetsCreateRowInfo(sheet_name, sheet_id, run_id,arch, split_type, 
                            h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                            num_params, batch_size, loss_module, num_decisions, limit_train, 
                            pooling_op, dropout):
    new_row_info = [run_id, 'Full' if not limit_train else limit_train, num_decisions, pooling_op, \
                    arch, num_params, split_type, h_SEED, batch_size, dropout, loss_module, val_acc, test_acc]
    header = ["RunID",	"Training Set", "#Decisions", 'Pooling Operation', \
        "Arch", '#Parameters', "SplitType","Seed", "Batch size", "Dropout","Loss func", 'Val Accuracy', 'Test Accuracy']
    return new_row_info, header

def writeToGoogleSheetsLinear(sheet_name, sheet_id, run_id,
                            arch, split_type, h_SEED, dict_double_val,dict_double_test, 
                            val_acc, test_acc, num_params, batch_size, loss_module, 
                            num_decisions, limit_train, pooling_op, dropout, use_drive=True):
    
    new_row_info, header = writeToGoogleSheetsCreateRowInfo(sheet_name, sheet_id, run_id,arch, 
                            split_type, h_SEED, dict_double_val,dict_double_test, 
                            val_acc, test_acc, num_params, batch_size, loss_module,  num_decisions, limit_train, 
                            pooling_op, dropout)
    
    header_names_val, header_names_test, new_row_val, new_row_test = writeToGoogleSheetsExtractResultsForRow(dict_double_val, dict_double_test)
    
    header = header + header_names_val+header_names_test
    new_row = new_row_info+new_row_val+new_row_test
    
    if use_drive:
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        print(last_row)
    
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        worksheet.insert_rows(last_row, number=1, values= new_row)
    else:
        #print(new_row.shape(), new_row.squeeze(1).shape)
        df = pd.DataFrame([new_row], columns=header)
        print(df)
        

def writeToGoogleSheetsSimpleCreateInfoRow(sheet_name, sheet_id, run_id, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, 
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                num_params, batch_size, loss_module, num_decisions, limit_train, group_embedding_dim, dropout, pooling_op=None):
    new_row_info = [run_id, 'Full' if not limit_train else limit_train, num_decisions]
    if pooling_op != None:
        new_row_info += [pooling_op]
    new_row_info += [h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, group_embedding_dim, num_params, split_type, h_SEED, batch_size, dropout, loss_module, val_acc, test_acc]
    header = ["RunID",	"Training Set", "#Decisions"]
    if pooling_op != None:
        header += ['Pooling Operation']
    header += ["MessageArchFirst",	"NodeArchFirst",	"MessageArchSecond",	"NodeArchSecond",	'GroupEmbedDim', '#Parameters', "SplitType"	,"Seed", "Batch size", "Dropout", "Loss func", 'Val Accuracy', 'Test Accuracy']
    return header, new_row_info

def writeToGoogleSheetsExtractResultsForRow(dict_double_val, dict_double_test):
    new_row_val = []
    dict_report = dict_double_val
    print(dict_report)
    header_names_val = []
    for key in dict_report:
        if type(dict_report[key]) == dict:
            for key_small in dict_report[key]:
                if 'macro avg' in key:
                    new_row_val.insert(0,dict_report[key][key_small])
                    header_names_val.insert(0,"val: "+key+":"+key_small)
                else:    
                    new_row_val.append(dict_report[key][key_small]) 
                    header_names_val.append("val: "+key+":"+key_small)
        else:
            new_row_val.append(dict_report[key]) 
            header_names_val.append("val: "+key)
    new_row_test = []
    dict_report = dict_double_test
    print(dict_report)
    header_names_test = []
    for key in dict_report:
        if type(dict_report[key]) == dict:
            for key_small in dict_report[key]:
                if 'macro avg' in key:
                    new_row_test.insert(0,dict_report[key][key_small])
                    header_names_test.insert(0,"test: "+key+":"+key_small)
                else:    
                    new_row_test.append(dict_report[key][key_small]) #, dict_report[key][key_small])
                    header_names_test.append("test: "+key+":"+key_small)
        else:
            new_row_test.append(dict_report[key])
            header_names_test.append("test: "+key)
    return header_names_val, header_names_test, new_row_val, new_row_test

def writeToGoogleSheetsSimpleCreateRow(sheet_name, sheet_id, run_id, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, 
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                num_params, batch_size, loss_module, num_decisions, limit_train, group_embedding_dim, dropout, pooling_op=None):
    header, new_row_info = writeToGoogleSheetsSimpleCreateInfoRow(sheet_name, sheet_id, run_id, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2,
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                num_params, batch_size, loss_module, num_decisions, limit_train, group_embedding_dim, dropout, pooling_op) 
    header_names_val, header_names_test, new_row_val, new_row_test = writeToGoogleSheetsExtractResultsForRow(dict_double_val, dict_double_test)
    
    header = header + header_names_val+header_names_test
    new_row = new_row_info+new_row_val+new_row_test
    print(new_row)
    return header, new_row

def writeToGoogleSheetsSimple(sheet_name, sheet_id, run_id, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, 
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                num_params, batch_size, loss_module,limit_train,num_decisions,group_embedding_dim, pooling_op, dropout, use_drive=True,path=None): 
    
    
    header, new_row = writeToGoogleSheetsSimpleCreateRow(sheet_name, sheet_id, run_id, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, 
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                num_params, batch_size, loss_module, num_decisions, limit_train, group_embedding_dim, dropout, pooling_op)
    if path != None:
        header.append('PathToBest')
        new_row.append(path)
        
    if use_drive:
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')
        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        print(last_row)
        print(header)
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        
        is_written = False
        while not is_written:
            worksheet.insert_rows(last_row, number=1, values= new_row)
            time.sleep(4)
            if checkIfWritten(sheet_name, sheet_id, run_id):
                is_written = True
    else:
        #print(new_row.shape(), new_row.squeeze(1).shape)
        df = pd.DataFrame([new_row], columns=header)
        print(df)


def writeToGoogleSheetsSimpleCrossVal(sheet_name, sheet_id, run_id, common_ident, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, 
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                num_params, batch_size, loss_module, num_decisions, limit_train, group_embedding_dim, pooling_op, dropout, use_drive=True):
    
    header, new_row = writeToGoogleSheetsSimpleCreateRow(sheet_name, sheet_id, run_id, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, 
                split_type, h_SEED, dict_double_val,dict_double_test, val_acc, test_acc, 
                num_params, batch_size, loss_module, num_decisions, limit_train, group_embedding_dim, dropout, pooling_op)
    if use_drive: 
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        print(last_row)
        print(header)
        header = ['Common ID']+header
        new_row = [common_ident]+new_row
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        worksheet.insert_rows(last_row, number=1, values= new_row)
    else:
        df = pd.DataFrame([new_row], columns=header)
        print(df)

def writeToGoogleSheetsCrossValSummary(sheet_name, sheet_id, data_over_folds, h_mess_arch_1, h_node_arch_1, h_mess_arch_2, h_node_arch_2, 
                split_type, h_SEED, batch_size, loss_module, limit_train, group_embedding_dim, pooling_op, dropout, use_drive=True):
    
    header = ["CommonID",	"Training Set", "#Decisions", "Pooling Operation",  \
            "MessageArchFirst",	"NodeArchFirst",	"MessageArchSecond", \
        	"NodeArchSecond",	"GroupEmbedDim",'#Parameters', "SplitType"	,"Seed", "Batch size", \
            "Dropout", "Loss func"]
    num_decision_str = str(np.mean(data_over_folds['num_decisions']))+' +- '+str(np.std(data_over_folds['num_decisions']))
    new_row = [data_over_folds['ident'], 'Full' if not limit_train else limit_train, num_decision_str, pooling_op,
                    h_mess_arch_1, h_node_arch_1, h_mess_arch_2, \
                    h_node_arch_2, group_embedding_dim, data_over_folds['num_params'], split_type, h_SEED, batch_size, \
                    dropout, loss_module]
    datasets = ['val', 'test']
    values = ['acc', 'macro avg: f1']
    for d in datasets:
        for v in values:
            m = np.mean(data_over_folds[d][v])
            sd = np.std(data_over_folds[d][v])
            header.append('(mean)'+d+': '+v)
            new_row.append(m)
            header.append('(sd)'+d+': '+v)
            new_row.append(sd)
    if use_drive:
        gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')

        #open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
        sh = gc.open(sheet_name)
        print(sh, sheet_id, type(sh))
        worksheet = sh[sheet_id]
        cells = worksheet.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        last_row = len(cells)
        print(last_row)
        print(header)
        if last_row == 1:
            worksheet.insert_rows(0, number=1, values= header)    
        #worksheet.insert_rows(last_row, number=1, values= new_row)
        is_written = False
        while not is_written:
            worksheet.insert_rows(last_row, number=1, values= new_row)
            time.sleep(4)
            if checkIfWritten(sheet_name, sheet_id, data_over_folds['ident']):
                is_written = True
    else:
        df = pd.DataFrame([new_row], columns=header)
        print(df)
#


# In[163]:
def get_df_from_worksheet_records(records, empty_value='', has_header=True, numerize=True):
    if len(records) == 0:
        return pd.DataFrame()
    keys = records[0].keys()
    values = []
    for dict_item in records:
        row = []
        print(dict_item)
        for key in  keys:
            row.append(dict_item[key])
        values.append(row)
    if has_header:
        keys = records[0].keys()
        if any(key == '' for key in keys):
            print('At least one column name in the data frame is an empty string. If this is a concern, please specify include_tailing_empty=False and/or ensure that each column containing data has a name.')
        df = pd.DataFrame(values, columns=keys)
    else:
        df = pd.DataFrame(values)

    return df


def getDoubleDict(test_result):
    dict_double = {}
    test_result_dict = test_result[0]
    for key in test_result_dict:
        if key == 'test_acc' or key == "val_acc":
            continue
        splitting = str(key).split('-')
        if len(splitting) > 2:
            #header.append(splitting[1])
            if splitting[1] not in dict_double:
                dict_double[splitting[1]] = {}
            dict_double[splitting[1]][splitting[2]] = test_result_dict[key]
        else:
            print(key, test_result_dict[key])
    return dict_double
