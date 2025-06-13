# Overview
This repo provides the code and links to the dataset as well as the data collection protocol used for producing the results of the paper <em>**Templates and Graph Neural Networks for Social Robots Interacting in Small Groups of Varying Sizes**</em> by [Sarah Gillet](https://sarahgillet.com/), [Sydney Thompson](https://www.sydneythompson.dev/), [Iolanda Leite](https://iolandaleite.com/) and [Marynel Vázquez](https://www.marynel.net/). 
Instructions on how to run the code above can be found below under "Instructions". The dataset and data collection protocol is documented under [docs/dataset](https://github.com/sarahgillet/TGM-SmallGroups/tree/main/docs/dataset.md) and further example commands can be found at [docs/commands](https://github.com/sarahgillet/TGM-SmallGroups/tree/main/docs/commands.md). 

# Instructions
1. Install docker. We use Docker version 20.10.17, build 100c701. You will need a dockerhub account to pull the base image. 

2. Login to your dockerhub account

3. unpack the zip and cd to the GNNGroupImitationLearning folder

4. Build the docker image through: docker build . --tag gnn_docker

5. Then run the following examp0lary command for testing. More commands based on the best model architectures as identified during our hyperparamter exploration are listed in [docs/commands](https://github.com/sarahgillet/TGM-SmallGroups/tree/main/docs/commands.md):

        docker run gnn_docker python3.8 ./brainstorming_data_train_who.py --do_test=True --project_config_name=BrainstormingExperimental --h_SEED=0 --early_stopping=True --tune_learning_rate=True --google_sheet_id=1 --GNN_second_layer=False --h_mess_arch_1='8-16-4' --h_node_arch_1='4-8' --h_mess_arch_2='-' --h_node_arch_2='-' --n_epochs=1000 --loss_module=MSELoss --h_BATCH_SIZE=128 --limit_train=False --dropout=0.2

# How to interpret the results?
First, the training will output the results for the training and validation on triads/dyads and then load the model again to output the result on the group of four. 

