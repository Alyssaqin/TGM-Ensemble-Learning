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
    early_stopping=False,
    tune_learning_rate=False,
    n_epochs=500,
    group_embedding_dim=6, # was 12 for all other runs
    project_config_name='Teenager-SelectedFeatures-ExploreFineTuning',
    google_sheet_id=3, # give ID of sheet starting with 1
    wandb_project_name="GNN-Teenager-Who-SelectedFeatures-ExploreFineTuning",
    GNN_second_layer=True,
    GNN_third_layer=False,
    h_mess_arch_1 = '4',#also could try it
    h_node_arch_1 = '2',
    h_mess_arch_2 = '4',
    h_node_arch_2 = '2',
    selected_indices="[0,1,2,3,4,5,6,7,8,9,10,11,12]",
    loss_module = 'MSELOSS',
    augment=True,
    mix_episodes = True,
    teen_group = 'WALLE',
    pretrained = True, 
    dropout=0.1,
    learning_rate=0.008,
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

def main(argv):

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
    selected_indices = [int(x) for x in FLAGS.selected_indices.split(',')]
    FLAGS.selected_indices = selected_indices
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

    model_paths_full = []
    model_paths_r2d2 = []
    model_paths_walle = []
    model_paths_icub = []
    
    FLAGS.learning_rate = 0.008
    dropout_rates = [0.01,0.185, 0.446]
    for dropout in dropout_rates:
        print(f"Starting training with dropout_rate: {dropout}")
        FLAGS.dropout = dropout

        path, val_result, test_result,test_loss, num_params, run_id_full,test_dataset = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'Full', FLAGS)
        dict_double_val = utils.getDoubleDict(val_result)
        dict_double_test = utils.getDoubleDict(test_result)

        print(f"Test Loss for lr={dropout}: {test_loss}")
        print(f"Test Acc for lr={dropout}: {test_result[0].get('test_acc', 'N/A')}")
        model_paths_full.append((path, val_result[0]['val_loss']))

        utils.writeToGoogleSheetsSimple(FLAGS.project_config_name, FLAGS.google_sheet_id-1, run_id_full, 
                            FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
                            'Mixed', FLAGS.h_SEED, dict_double_val, dict_double_test, 
                            val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                            FLAGS.h_BATCH_SIZE, FLAGS.loss_module, ','.join([str(index) for index in FLAGS.selected_indices]),
                            f'Full_dropout{dropout}',
                            FLAGS.group_embedding_dim, 'min', 'not applicable', 
                            path=path)  

        utils.set_random_seed(FLAGS.h_SEED)
        path_r2d2, val_result, test_result, test_loss,num_params, run_id_r2d2,test_dataset_r2d2 = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'R2D2', FLAGS, finetune=True, path=path)
        dict_double_val = utils.getDoubleDict(val_result)
        dict_double_test = utils.getDoubleDict(test_result)
        model_paths_r2d2.append((path_r2d2, val_result[0]['val_loss']))

        utils.writeToGoogleSheetsSimpleCrossVal(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id_r2d2, run_id_full,
                                FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
                                'Mixed', FLAGS.h_SEED, dict_double_val, dict_double_test, 
                                val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                                FLAGS.h_BATCH_SIZE, FLAGS.loss_module, ','.join([str(index) for index in FLAGS.selected_indices]),
                                f'R2D2_dropout{dropout}',
                                FLAGS.group_embedding_dim, 'min', path_r2d2)                          
        utils.set_random_seed(FLAGS.h_SEED)
        path_walle, val_result, test_result, test_loss,num_params, run_id_walle,test_dataset_walle = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'WALLE', FLAGS, finetune=True, path=path)
        dict_double_val = utils.getDoubleDict(val_result)
        dict_double_test = utils.getDoubleDict(test_result)
        model_paths_walle.append((path_walle, val_result[0]['val_loss']))

        utils.writeToGoogleSheetsSimpleCrossVal(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id_walle, run_id_full,
                                FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
                                'Mixed', FLAGS.h_SEED, dict_double_val, dict_double_test, 
                                val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                                FLAGS.h_BATCH_SIZE, FLAGS.loss_module,  ','.join([str(index) for index in FLAGS.selected_indices]),
                                f'WALLE_dropout{dropout}',
                                FLAGS.group_embedding_dim, 'min', path_walle)   

        utils.set_random_seed(FLAGS.h_SEED)
        path_icub, val_result, test_result,test_loss, num_params, run_id_icub,test_dataset_icub = get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, 'ICUB', FLAGS, finetune=True, path=path)
        dict_double_val = utils.getDoubleDict(val_result)
        dict_double_test = utils.getDoubleDict(test_result)
        model_paths_icub.append((path_icub, val_result[0]['val_loss']))

        utils.writeToGoogleSheetsSimpleCrossVal(FLAGS.project_config_name, FLAGS.google_sheet_id, run_id_icub, run_id_full,
                                FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, 
                                'Mixed', FLAGS.h_SEED, dict_double_val, dict_double_test, 
                                val_result[0]['val_acc'], test_result[0]['test_acc'], num_params, 
                                FLAGS.h_BATCH_SIZE, FLAGS.loss_module,  ','.join([str(index) for index in FLAGS.selected_indices]),
                                f'ICUB_dropout{dropout}',
                                FLAGS.group_embedding_dim, 'min',path_icub)   

if __name__ == '__main__':
    absl.app.run(main)
