import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from tqdm import tqdm
import os
import pygsheets
import time


import MMPN
from dataset import SummerschoolDataset  
from argparse import Namespace
import pygsheets
import config
import utils
from sklearn.metrics import f1_score

def get_model_paths_from_gsheet(sheet_name="Teenager", sheet_id=2, top_k=3):
    import pygsheets
    import pandas as pd
    import os

    gc = pygsheets.authorize(service_file="/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json")
    sh = gc.open(sheet_name)
    worksheet = sh[sheet_id]

    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    df["val: macro avg:f1"] = pd.to_numeric(df["val: macro avg:f1"], errors='coerce')

    top_df = df.sort_values(by="val: macro avg:f1", ascending=False).head(top_k)

    selected_models = []
    for _, row in top_df.iterrows():
        ckpt_relative_path = row["PathToBest"] 
        ckpt_absolute_path = os.path.abspath(os.path.join(os.getcwd(), ckpt_relative_path))
        tag = row["#Decisions"] if "#Decisions" in row else row["Training Set"]
        selected_models.append((ckpt_absolute_path, tag))

    return selected_models

model_info = get_model_paths_from_gsheet(sheet_name="Teenager", sheet_id=2, top_k=3)
print("Selected Top-3 models by val macro F1:")
for path, tag in model_info:
    print(f"Tag: {tag}, Path: {path}")

def patch_predict_softmax(model):
    def predict_softmax(self, batch):
        x, edge_index = batch.x, batch.edge_index
        global_attr, batch_idx = batch.global_attr, batch.batch
        x = model.model(x, edge_index, global_attr, 3, 6, batch_idx)
        return F.softmax(x, dim=1)
    model.predict_softmax = types.MethodType(predict_softmax, model)


def confidence_dynamic_ensemble_predict(batch, models):
    with torch.no_grad():
        softmax_outputs = []
        confidences = []

        for model in models:
            probs = model.predict_softmax(batch)
            softmax_outputs.append(probs)
            conf = probs.max(dim=1).values  
            confidences.append(conf.unsqueeze(1))  

        weighted_output = torch.zeros_like(softmax_outputs[0])
        total_confidence = torch.zeros_like(confidences[0])

        for prob, conf in zip(softmax_outputs, confidences):
            weighted_output += prob * conf
            total_confidence += conf

        total_confidence = torch.clamp(total_confidence, min=1e-6) 
        weighted_output = weighted_output / total_confidence

        preds = weighted_output.argmax(dim=1)

        if hasattr(batch, 'y_who'):
            acc = (preds == batch.y_who).sum().float() / preds.shape[0]
            return preds, acc, weighted_output
        return preds, None, weighted_output
    
def evaluate_ensemble(predict_fn, dataloader, group_name=""):
    total_correct = 0
    total_samples = 0
    for batch in tqdm(dataloader, desc=f"Evaluating {group_name}"):
        batch = batch.to(FLAGS.device)
        preds, acc, _ = predict_fn(batch)
        total_correct += (preds == batch.y_who).sum().item()
        total_samples += preds.shape[0]
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"[{group_name}] Confidence Ensemble Accuracy: {round(overall_acc, 4)}")
    return overall_acc

def evaluate_ensemble_f1(predict_fn, dataloader):
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader):
        batch = batch.to(FLAGS.device)
        preds, _, _ = predict_fn(batch)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y_who.cpu().tolist())
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Ensemble Macro-F1: {round(macro_f1, 4)}")
    return macro_f1

def create_test_loader_for_group(group_name):
    dataset = SummerschoolDataset(
        "../training_data",
        "allepisodes_norm_no_permutations_norm.csv",
        lookback=FLAGS.h_lookback,
        kept_indices=kept_indices_manual_selection,
        name_filter='manual_' + '_'.join(selected_features),
        use_samples=FLAGS.use_samples,
        augment_mfcc=FLAGS.augment
    )
    _, _, test_dataset, _ = utils.generateTrainDoubleTestValidDataset(
        dataset, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, fixed_group=group_name)
    _, _, test_loader = utils.loadData(None, test_dataset, test_dataset, FLAGS.h_BATCH_SIZE)
    return test_loader

def create_val_loader_for_group(group_name):
    dataset = SummerschoolDataset(
        "../training_data",
        "allepisodes_norm_no_permutations_norm.csv",
        lookback=FLAGS.h_lookback,
        kept_indices=kept_indices_manual_selection,
        name_filter='manual_' + '_'.join(selected_features),
        use_samples=FLAGS.use_samples,
        augment_mfcc=FLAGS.augment
    )
    _, val_dataset, _, _ = utils.generateTrainDoubleTestValidDataset(
        dataset, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, fixed_group=group_name)
    _, val_loader, _ = utils.loadData(None, val_dataset, val_dataset, FLAGS.h_BATCH_SIZE)
    return val_loader

def write_results_to_google_sheet(sheet_name, sheet_id, result_dict):
    import pygsheets

    gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')
    sh = gc.open(sheet_name)
    worksheet = sh[sheet_id]

    header = ['Group', 'Val Macro-F1', 'Val Accuracy', 'Test Macro-F1', 'Test Accuracy']

    
    first_cell = worksheet.get_value('A1')
    if first_cell != 'Group':
        worksheet.insert_rows(0, number=1, values=header)

    rows = []
    for group, metrics in result_dict.items():
        row = [
            group,
            round(metrics['val_f1'], 4),
            round(metrics['val_acc'], 4),
            round(metrics['test_f1'], 4),
            round(metrics['test_acc'], 4)
        ]
        rows.append(row)

    for row in rows:
        worksheet.append_table(row, start='A1', dimension='ROWS', overwrite=False)


if __name__ == "__main__":

    FLAGS = Namespace(
        device='cpu',
        h_lookback=20,
        use_samples=1,
        h_TEST_SIZE=0.06,
        h_VALID_SIZE=0.1,
        h_BATCH_SIZE=64,
        h_SEED=0,
        memory_type='LSTM',
        early_stopping=False,
        tune_learning_rate=False,
        n_epochs=500,
        group_embedding_dim=6,
        project_config_name='Teenager-SelectedFeatures-ExploreFineTuning',
        google_sheet_id=3,
        wandb_project_name="GNN-Teenager-Who-SelectedFeatures-ExploreFineTuning",
        GNN_second_layer=True,
        GNN_third_layer=False,
        h_mess_arch_1='4',
        h_node_arch_1='2',
        h_mess_arch_2='4',
        h_node_arch_2='2',
        selected_indices="[0,1,2,3,4,5,6,7,8,9,10,11,12]",
        loss_module='MSELOSS',
        augment=True,
        mix_episodes=True,
        teen_group='WALLE',
        pretrained=True,
        learning_rate=0.005,
        pretrained_path='',
        dropout=0.1  
    )

    selected_features = ['mfcc_Std', 'energy_Std', 'pitch_Std', '1stEnergy_Std', '1stPitch_Std']
    selected_indices = [4, 5, 9, 10, 12]
    features_selected_temp = [list(config.FEATURE_DICT.keys())[i] for i in selected_indices]

    feature_sel_dict_init = {k: False for k in config.FEATURE_DICT}
    feature_sel_dict, kept_indices_manual_selection = utils.createFeatureBoolDict(
        features_selected_temp,
        feature_sel_dict_init,
        config.FEATURE_DICT
    )

    dataset = SummerschoolDataset(
        "../training_data",
        "allepisodes_norm_no_permutations_norm.csv",
        lookback=FLAGS.h_lookback,
        kept_indices=kept_indices_manual_selection,
        name_filter='manual_' + '_'.join(selected_features),
        use_samples=FLAGS.use_samples,
        augment_mfcc=FLAGS.augment
    )

    _, val_dataset, test_dataset, _ = utils.generateTrainDoubleTestValidDataset(
    dataset, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE)
    _, val_loader, _ = utils.loadData(None, val_dataset, val_dataset, FLAGS.h_BATCH_SIZE)
    _, _, test_loader = utils.loadData(None, test_dataset, test_dataset, FLAGS.h_BATCH_SIZE)


    model_paths = get_model_paths_from_gsheet(sheet_name="Teenager", sheet_id=2, top_k=3)

    models = []
    for path, _ in model_paths:
        model = MMPN.NodeLevelMMPN.load_from_checkpoint(path)
        model.to(FLAGS.device)
        model.eval()
        patch_predict_softmax(model)
        models.append(model)

  
    predict_fn = lambda batch: confidence_dynamic_ensemble_predict(batch, models)

    groups = ["full", "R2D2", "WALLE", "ICUB"]
    result_dict = {}

    for group in groups:
        print(f"\n--- Evaluating group: {group} ---")

        if group == "full":
            val_loader_group = val_loader
            test_loader_group = test_loader
        else:
            val_loader_group = create_val_loader_for_group(group)
            test_loader_group = create_test_loader_for_group(group)

     
        val_f1 = evaluate_ensemble_f1(predict_fn, val_loader_group)
        val_acc = evaluate_ensemble(predict_fn, val_loader_group)

       
        test_f1 = evaluate_ensemble_f1(predict_fn, test_loader_group)
        test_acc = evaluate_ensemble(predict_fn, test_loader_group)

        result_dict[group] = {
            "val_f1": val_f1,
            "val_acc": val_acc,
            "test_f1": test_f1,
            "test_acc": test_acc
        }

    write_results_to_google_sheet(
        sheet_name="Result",
        sheet_id=9,
        result_dict=result_dict
    )

    





