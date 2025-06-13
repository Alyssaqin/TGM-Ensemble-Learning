# Exemplary commands
## Note on commands 
We only provide exemplary commands for best results that we received. You are invited to explore different hyperparameter settings or run different entry points to the training. All models for the brainstorming dataset were trained using one of the brainstorming_* python scripts. 

If you download the github repository as is, the training is based on processed data files. We also released the raw comma-separated files for the [brainstorming dataset](https://huggingface.co/datasets/sarahgillet/BrainstormingDataset). More information is also provided under [docs/dataset](https://github.com/sarahgillet/TGM-SmallGroups/tree/main/docs/dataset.md) The teenager dataset could not be provided at this point. We provide all scripts used for training but the training cannot be run with the files we provided. We are discussing with the original authors about possibilities for releasing the dataset.

## Commands for training on the full brainstorming dataset, who: 
#### TGM
    docker run gnn_docker python3.8 ./brainstorming_data_train_who.py --do_test=True --project_config_name=BrainstormingExperimental --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --GNN_second_layer=False --h_mess_arch_1='8-16-4' --h_node_arch_1='4-8' --h_mess_arch_2='-' --h_node_arch_2='-' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=128 --limit_train=False --dropout=0.2

#### MLP
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_linear.py -do_test=True --limit_train=False --project_config_name=BrainstormingMLP --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --arch='8-16' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --dropout=0.2

#### RF
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_linear_trad_RF.py --do_test=True --n_estimators=100 --max_depth=10 --min_samples_split=2 --min_samples_leaf=2 --max_features='sqrt' --bootstrap=False --criterion='gini' --h_SEED=0

## Commands for training on the full brainstorming dataset, what:
#### TGM
    docker run gnn_docker python3.8 ./brainstorming_data_train_what.py --do_test=True --project_config_name=BrainstormingExperimental_what --h_SEED=5 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --GNN_second_layer=True --h_mess_arch_1='8-16' --h_node_arch_1='8-16' --h_mess_arch_2='8-16' --h_node_arch_2='4-8' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --limit_train=False --dropout=0.2

#### MLP
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_linear.py --do_test=True --limit_train=False --project_config_name=BrainstormingMLP_what --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --arch='4-16' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --dropout=0.2

#### RF
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_linear_trad_RF.py --do_test=True --n_estimators=100 --max_depth=30 --min_samples_split=2 --min_samples_leaf=2 --max_features='sqrt' --bootstrap=False --criterion='entropy' --h_SEED=0 

## Commands for training on the subsets of the brainstorming dataset, who:
Note that the following examples take longer to compute than the ones above because they run 6-fold cross validation.
### Training set: 'Dyads'
#### TGM
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_cross_val.py --do_test=True --limit_train=True --group_size_training='2' --project_config_name=BrainstormingExperimental_subset --h_SEED=2 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --GNN_second_layer=True --h_mess_arch_1='8-16' --h_node_arch_1='4-8' --h_mess_arch_2='4-8' --h_node_arch_2='4-8' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --dropout=0.2

#### MLP
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_linear_cross_val.py --do_test=True --limit_train=True --group_size_training='2' --project_config_name=BrainstormingMLP_subset --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --arch='4-8' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --pooling_operation='min'

#### RF
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_linear_cross_val_trad.py --n_estimators= 100 --max_depth=30 --min_samples_split=2 --min_samples_leaf=1 --max_features=sqrt --bootstrap=False --criterion='entropy' --h_SEED=0 --group_size_training='2' --limit_train=True --do_test=True

### Training set: 'Triads'
#### TGM
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_cross_val.py --do_test=True --limit_train=True --group_size_training='3' --project_config_name=BrainstormingExperimental_subset --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=4 --GNN_second_layer=False --h_mess_arch_1='8-16' --h_node_arch_1='16-4' --h_mess_arch_2=- --h_node_arch_2=- --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --dropout=0.2

#### MLP
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_linear_cross_val.py --do_test=True --limit_train=True --group_size_training='3' --project_config_name=BrainstormingMLP_subset --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=4 --arch='4-16' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --pooling_operation='min'

#### RF
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_linear_cross_val_trad.py --n_estimators=500 --max_depth=30 --min_samples_split=2 --min_samples_leaf=1 --max_features='sqrt' --bootstrap=False --criterion='entropy' --h_SEED=0 --group_size_training='3' --limit_train=True --do_test=True


### Training set: 'Mixed'
#### TGM
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_cross_val.py --do_test=True --limit_train=True --group_size_training='2-3' --project_config_name=BrainstormingExperimental_subset --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=7 --GNN_second_layer=True --h_mess_arch_1='4-8' --h_node_arch_1='4-8' --h_mess_arch_2='16-4' --h_node_arch_2='16-4' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --dropout=0.2

#### MLP
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_linear_cross_val.py --do_test=True --limit_train=True --group_size_training='2-3' --project_config_name=BrainstormingMLP_subset --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=7 --arch='16-4' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=64 --pooling_operation='min'

#### RF
    docker run gnn_docker python3.8 ./brainstorming_data_train_who_linear_cross_val_trad.py --n_estimators=400 --max_depth=40 --min_samples_split=2 --min_samples_leaf=1 --max_features='sqrt' --bootstrap=False --criterion='entropy' --h_SEED=0 --group_size_training='2-3' --limit_train=True --do_test=True


## Commands for training on the subsets of the brainstorming dataset, what:
### Training set: 'Dyads'
#### TGM
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_cross_val.py --do_test=True --limit_train=True --group_size_training='2' --project_config_name=BrainstormingExperimental_subset_what --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --GNN_second_layer=True --h_mess_arch_1='4-8' --h_node_arch_1='4-8' --h_mess_arch_2='4-8' --h_node_arch_2='8-16' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=32 --dropout=0.5

#### MLP
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_linear_cross_val.py --do_test=True --limit_train=True --group_size_training='2' --project_config_name=BrainstormingMLP_subset_what --h_SEED=2 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --arch='4-16' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=32 --pooling_operation='min'

#### RF
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_linear_cross_val_trad_RF.py --n_estimators=100  --max_depth=10 --min_samples_split=2 --min_samples_leaf=2 --max_features='sqrt' --bootstrap=False --criterion='gini' --h_SEED=0 --group_size_training='2' --limit_train=True

### Training set: 'Triads'
#### TGM
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_cross_val.py --do_test=True --limit_train=True --group_size_training='3' --project_config_name=BrainstormingExperimental_subset_what --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=4 --GNN_second_layer=False --h_mess_arch_1='4-8' --h_node_arch_1='4-8' --h_mess_arch_2=- --h_node_arch_2=- --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=32 --dropout=0.2

#### MLP
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_linear_cross_val.py --do_test=True --limit_train=True --group_size_training='3' --project_config_name=BrainstormingMLP_subset_what --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --arch='8-16' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=32 --pooling_operation='min'

#### RF
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_linear_cross_val_trad_RF.py --n_estimators=400  --max_depth=10 --min_samples_split=2 --min_samples_leaf=1 --max_features='auto' --bootstrap=True --criterion='gini' --h_SEED=0 --group_size_training='3' --limit_train=True

### Training set: 'Mixed'
#### TGM
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_cross_val.py --do_test=True --limit_train=True --group_size_training='2-3' --project_config_name=BrainstormingExperimental_subset_what --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=7 --GNN_second_layer=True --h_mess_arch_1='4-8' --h_node_arch_1='4-8' --h_mess_arch_2='16-4' --h_node_arch_2='8-16' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=32 --dropout=0.5

#### MLP
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_linear_cross_val.py --do_test=True --limit_train=True --group_size_training='2-3' --project_config_name=BrainstormingMLP_subset_what --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --arch='4-8' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=32 --pooling_operation='min'

#### RF
    docker run gnn_docker python3.8 ./brainstorming_data_train_what_linear_cross_val_trad_RF.py --n_estimators=100  --max_depth=10 --min_samples_split=5 --min_samples_leaf=1 --max_features=sqrt --bootstrap=False --criterion='entropy' --h_SEED=0 --group_size_training='2-3' --limit_train=True





