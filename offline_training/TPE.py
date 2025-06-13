#!/usr/bin/env python
# coding: utf-8

# In[137]:


import utils
import config
import wandb
import torch
from dataset import SummerschoolDataset
import MMPN
import torch.nn as nn

import absl.app
import absl.flags
import optuna


utils.set_random_seed(42)


FLAGS_DEF = utils.define_flags_with_default(
    device= 'cpu',
    h_lookback= 20,
    use_samples=1,
    h_TEST_SIZE= 0.06,
    h_VALID_SIZE= 0.1,
    h_BATCH_SIZE = 64,
    h_SEED = 0,
    memory_type='LSTM',
    early_stopping=True,
    tune_learning_rate=False,
    n_epochs=500,
    group_embedding_dim=6, # was 12 for all other runs
    project_config_name='Teenager-SelectedFeatures-ExploreFineTuning',
    google_sheet_id=3, # give ID of sheet starting with 1
    wandb_project_name="GNN-Teenager-Who-SelectedFeatures-ExploreFineTuning",
    GNN_second_layer=True,
    GNN_third_layer=False,
    h_mess_arch_1 = '4',
    h_node_arch_1 = '2',
    h_mess_arch_2 = '4',
    h_node_arch_2 = '2',
    selected_indices="[0,1,2,3,4,5,6,7,8,9,10,11,12]",
    loss_module = 'MSELOSS',
    augment=True,
    mix_episodes = True,
    teen_group = 'WALLE',
    pretrained = True, 
    learning_rate=0.005,
    dropout = 0.2,
    pretrained_path = '/home/rpl/GNNImitationLearning/GNNGroupImitationLearning/checkpoints/GraphLevelNodeLevelMMPN/643de42ed1d4437b8ea192e8e39826a4/epoch=881-val_loss=0.18800-val_acc=0.40371.ckpt'
)

def get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, training_dataset, FLAGS, finetune=False, path=''):
    # if training_dataset=='Full':
    #     training_dataset=None
    train_dataset, test_dataset, valid_dataset, test_dataset_gen = utils.generateTrainDoubleTestValidDataset(dataset_org, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, fixed_group=training_dataset if training_dataset!='Full' else None)
    split_type = "double_test_ep_mixed"
    
    print(len(train_dataset.data.y_who),len(test_dataset.data.y_who), len(valid_dataset.data.y_who) , len(test_dataset_gen.data.y_who) )
    train_dataset = utils.upsampleTrain(train_dataset, 4, lambda x: x.y_who )
    print(len(train_dataset.data.y_who))
    graph_train_loader, graph_val_loader, graph_test_loader = utils.loadData(train_dataset, valid_dataset, test_dataset, FLAGS.h_BATCH_SIZE)
    model, result, val_result, test_result, test_loss, run_id, num_params, path  = utils.train_graph_classifier(model_name="NodeLevelMMPN", wandb_project_name=FLAGS.wandb_project_name, 
                                        h_SEED=FLAGS.h_SEED, device=FLAGS.device,
                                        graph_train_loader=graph_train_loader, graph_val_loader=graph_val_loader,
                                        graph_test_loader=graph_test_loader,
                                        early_stopping=FLAGS.early_stopping,
                                        tune_lr=FLAGS.tune_learning_rate,
                                        return_path = True,
                                        load_pretrained=finetune,
                                        pretrained_filename=path,
                                        train_pretrained=finetune,
                                        max_epochs=FLAGS.n_epochs,
                                        n_features_nodes = sum(kept_indices_manual_selection), 
                                        n_features_global = config.NUM_GLOB_FEATURES,
                                        message_arch = FLAGS.h_mess_arch_1, 
                                        node_arch = FLAGS.h_node_arch_1, 
                                        n_embedding_group = FLAGS.group_embedding_dim, 
                                        n_output_dim_node = 1, 
                                        n_output_dim_action = 1, 
                                        use_scheduler=True,
                                        second_layer=FLAGS.GNN_second_layer, 
                                        second_message_arch=FLAGS.h_mess_arch_2, 
                                        second_node_update_arch=FLAGS.h_node_arch_2, 
                                        split_mode = split_type, 
                                        memory_network_block = nn.LSTM if FLAGS.memory_type=='LSTM' else nn.GRU,
                                        lr=FLAGS.learning_rate,
                                        dropout=FLAGS.dropout,
                                        teen_group=FLAGS.teen_group,
                                        feat_sel=feature_sel_dict)
    
    #utils.writeToGoogleSheets(FLAGS.project_config_name, FLAGS.google_sheet_id-1, run_id, FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, split_type, FLAGS.h_lookback, FLAGS.h_SEED, feature_sel_dict, dict_double, dict_test, dataset=training_dataset)                                      
    wandb.finish()
    return path, val_result, test_result, test_loss, num_params, run_id,test_dataset


def objective(trial):
        
        feature_sel_dict = {
        'speechAmount': False, 
        'isSpeaking': False, 
        'loudness':False, 
        'mfcc': False, 
        'energy': False, 
        'pitch': False, 
        '1stEnergy': False, 
        '1stPitch': False, 
        'mfcc_Std': False, 
        'energy_Std': False, 
        'pitch_Std': False, 
        '1stEnergy_Std': False, 
        '1stPitch_Std': False
    }
    
        FLAGS = absl.flags.FLAGS
        FLAGS.n_epochs = 10
        FLAGS.learning_rate = trial.suggest_float("lr", 1e-5, 1e-2)
        FLAGS.dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
        if isinstance(FLAGS.selected_indices, str):
            selected_indices = [int(x) for x in FLAGS.selected_indices.strip("[]").split(',')]
        else:
            selected_indices = FLAGS.selected_indices

        #selected_indices = [int(x) for x in FLAGS.selected_indices.strip("[]").split(',')]
        #FLAGS.selected_indices = selected_indices
        #FLAGS.logging.project=FLAGS.project_config_name

        variant = utils.get_user_flags(FLAGS, FLAGS_DEF)
        print(variant)
        # Hyperparameters


        # Prepare the dataset
        utils.set_random_seed(FLAGS.h_SEED)
        # mixing_episodes is the parameter that determines if we split train test valid data by episodes or in the whole dataset
        

        # In[164]:

        print("Starting training")
        list_features = list(config.FEATURE_DICT.keys())
        features_selected_temp = []
        for feature_id in selected_indices:
            features_selected_temp.append(list_features[feature_id])
        print("Running on", features_selected_temp)
        feature_sel_dict, kept_indices_manual_selection = utils.createFeatureBoolDict(features_selected_temp, feature_sel_dict, config.FEATURE_DICT)
        
        manual_name_filter = 'manual_'+'_'.join(features_selected_temp)
        #manual_name_filter = 'norm_manual_'+'_'.join(features_selected_temp)
        print(manual_name_filter)
        dataset_org=SummerschoolDataset("../training_data", "allepisodes_norm_no_permutations_norm.csv", 
        #dataset_org=SummerschoolDataset("../training_data", "allepisodes_t_t_no_permutations_norm.csv", 
                                        lookback=FLAGS.h_lookback, 
                                        kept_indices=kept_indices_manual_selection, 
                                        name_filter=manual_name_filter, 
                                        use_samples=FLAGS.use_samples,
                                        augment_mfcc=FLAGS.augment)
        utils.set_random_seed(FLAGS.h_SEED)

        _, val_result, *_ = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'Full', FLAGS)
        val_acc=trial.set_user_attr("val_acc", val_result[0]["val_acc"])
        val_loss=trial.set_user_attr("val_loss", val_result[0]["val_loss"])


        return val_acc,val_loss
            
from absl import flags

if __name__ == '__main__':
    flags.FLAGS([''])
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=20)

    top5 = sorted(
        [t for t in study.trials if "val_acc" in t.user_attrs],
        key=lambda t: t.user_attrs["val_acc"],
        reverse=True
    )[:5]

    print("\n===== Top 5 Trials by val_acc =====")
    for i, trial in enumerate(top5, 1):
        print(f"\nTop {i}:")
        print(f"  val_acc = {trial.user_attrs['val_acc']:.4f}")
        print(f"  params = {trial.params}")
