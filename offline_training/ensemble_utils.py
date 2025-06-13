import utils
import torch.nn as nn
import torch.nn.functional as F
import config

def get_data_and_train(dataset_org, kept_indices_manual_selection, feature_sel_dict, training_dataset, FLAGS, finetune=False, path=''):
    # if training_dataset=='Full':
    #     training_dataset=None
    train_dataset, test_dataset, valid_dataset, test_dataset_gen = utils.generateTrainDoubleTestValidDataset(dataset_org, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, fixed_group=training_dataset if training_dataset!='Full' else None)
    split_type = "double_test_ep_mixed"
    
    print(len(train_dataset.data.y_who),len(test_dataset.data.y_who), len(valid_dataset.data.y_who) , len(test_dataset_gen.data.y_who) )
    num_classes = 4 
    if not hasattr(train_dataset.data, 'y_who_one'):
        train_dataset.data.y_who_one = F.one_hot(train_dataset.data.y_who, num_classes=num_classes).float()

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
                                        teen_group=FLAGS.teen_group,
                                        feat_sel=feature_sel_dict)
    
    #utils.writeToGoogleSheets(FLAGS.project_config_name, FLAGS.google_sheet_id-1, run_id, FLAGS.h_mess_arch_1, FLAGS.h_node_arch_1, FLAGS.h_mess_arch_2, FLAGS.h_node_arch_2, split_type, FLAGS.h_lookback, FLAGS.h_SEED, feature_sel_dict, dict_double, dict_test, dataset=training_dataset)                                      

    return path, val_result, test_result, test_loss, num_params, run_id, test_dataset, valid_dataset

