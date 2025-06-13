# Overview
This repository contains the code and experiments for my master's thesis project:  
**"Ensemble Learning for Social Robots: Enhancing Decision-Making in Human Group Interactions."**

This work builds upon the framework and dataset from the paper  
 *Templates and Graph Neural Networks for Social Robots Interacting in Small Groups of Varying Sizes*  
by [Sarah Gillet](https://github.com/sarahgillet), Sydney Thompson, Iolanda Leite, and Marynel Vázquez.  
[Original Paper Codebase (TGM-SmallGroups)](https://github.com/sarahgillet/TGM-SmallGroups)

# Introduction
Social robots interacting in groups face challenges due to the diversity of human behavior patterns.  
While the original **TGM + MPGNN** model performs well in specific settings, it struggles to generalize across different group structures, as observed in the Teenager dataset.

To address this, we propose an **ensemble learning approach** to enhance decision-making robustness, especially in the **"whom to address"** module of the TGM framework.

# Instructions
1. Install docker. We use Docker version 20.10.17, build 100c701. You will need a dockerhub account to pull the base image. 

2. Login to your dockerhub account

3. unpack the zip and cd to the GNNGroupImitationLearning folder

4. Build the docker image through: docker build . --tag gnn_docker

5. Then run the following examp0lary command for testing.
Here is for varaints training:
<pre><code>sudo docker run --rm \ -v /path/to/your_repo/offline_training:/usr/local/offline_training \-v /path/to/your_repo/training_data:/usr/local/training_data \-v /path/to/your_repo/checkpoints:/usr/local/checkpoints \-v /path/to/your_repo/offline_training/apikey/your-key.json:/usr/local/offline_training/apikey/your-key.json \-e GOOGLE_APPLICATION_CREDENTIALS=/usr/local/offline_training/apikey/your-key.json \-e PYTHONPATH=/usr/local/offline_training \-w /usr/local/offline_training gnn_docker \python3.8 TeenagerDataset_fine_tuning_ori.py \--project_config_name=Teenager \--h_SEED=5 \--early_stopping=True \--tune_learning_rate=False \--google_sheet_id=3 \--h_lookback=30 \--h_mess_arch_1=32 \--h_node_arch_1=8 \--GNN_second_layer=True \--h_mess_arch_2=8 \--h_node_arch_2=8 \--n_epochs=500 \--loss_module=MSELoss \--h_BATCH_SIZE=64 \--selected_indices='4,5,9,10,12'</code></pre>

Here is for fusion: 
<pre><code>sudo docker run --rm \-v /home/soro-student/TGM-SmallGroups-main/offline_training:/usr/local/offline_training \-v /home/soro-student/TGM-SmallGroups-main/training_data:/usr/local/training_data \-v /home/soro-student/TGM-SmallGroups-main/checkpoints:/usr/local/checkpoints \-v /home/soro-student/TGM-SmallGroups-main/offline_training/apikey/teenagers-450219-36bb8173156a.json:/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json \-e GOOGLE_APPLICATION_CREDENTIALS=/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json \-e PYTHONPATH=/usr/local/offline_training \-w /usr/local/offline_training \gnn_docker \python3 confidence.py</code></pre>

# Note
1. Replace /path/to/your_repo/ with the absolute path to your local repository.
2. Make sure to mount the /checkpoints folder, as it stores the trained model checkpoints that will be used later in the ensemble fusion process.
3. The apikey is a Google Cloud credential JSON used to access Google Sheets. You must provide your own service account key.

# How to interpret the results?
First, the training will output the results for the training and validation on triads/dyads and then load the model again to output the result on the group of four. 

