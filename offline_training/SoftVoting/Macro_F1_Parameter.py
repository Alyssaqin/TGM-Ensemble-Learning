import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from tqdm import tqdm
import os
import time 


import MMPN
from dataset import SummerschoolDataset  
from argparse import Namespace
import pygsheets
import config
import utils
from sklearn.metrics import f1_score

model_paths = [
         ('/usr/local/checkpoints/GraphLevelNodeLevelMMPN/e1d2c872aa5f48159d16b3d870181b41/epoch=249-val_loss=0.18698-val_acc=0.35377.ckpt','seed3'),
         ('/usr/local/checkpoints/GraphLevelNodeLevelMMPN/52c0ad2a898b4dd99fa2e8168d9e0453/epoch=35-val_loss=0.19694-val_acc=0.29717.ckpt','seed4'),
         ('/usr/local/checkpoints/GraphLevelNodeLevelMMPN/83f9a414a1c24c488d4e4e7cebf317b9/epoch=141-val_loss=0.19594-val_acc=0.27103.ckpt','seed5'),
    ]


def patch_predict_softmax(model):   
    def predict_softmax(self, batch):
        x, edge_index = batch.x, batch.edge_index
        global_attr, batch_idx = batch.global_attr, batch.batch
        x = model.model(x, edge_index, global_attr, 3, 6, batch_idx)
        return F.softmax(x, dim=1)
    model.predict_softmax = types.MethodType(predict_softmax, model)


def evaluate_val_f1_for_model(model, val_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model.predict_softmax(batch).argmax(dim=1).cpu()
            targets = batch.y_who.cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
    return f1_score(all_targets, all_preds, average='macro') if all_targets else 0.0


def compute_normalized_f1_weights(models, val_loader, device='cpu', eps=1e-4):
    f1_scores = [evaluate_val_f1_for_model(model, val_loader, device) for model in models]
    f1_tensor = torch.tensor(f1_scores)
    f1_tensor = torch.clamp(f1_tensor, min=eps) 
    normalized_weights = torch.softmax(f1_tensor, dim=0)
    return normalized_weights


def evaluate_ensemble(predict_fn, dataloader, group_name=""):
    total_correct = 0
    total_samples = 0
    for batch in tqdm(dataloader, desc=f"Evaluating {group_name}"):
        batch = batch.to(FLAGS.device)
        preds, acc, _ = predict_fn(batch)
        total_correct += (preds == batch.y_who).sum().item()
        total_samples += preds.shape[0]
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"[{group_name}] Ensemble Accuracy: {round(overall_acc, 4)}")
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

def ensemble_predict(batch, models, normalized_weights):
    with torch.no_grad():
        softmax_outputs = [model.predict_softmax(batch) for model in models]
        weighted_output = torch.zeros_like(softmax_outputs[0])
        for i, output in enumerate(softmax_outputs):
            weighted_output += normalized_weights[i] * output
        preds = weighted_output.argmax(dim=1)
        acc = None
        if hasattr(batch, 'y_who'):
            acc = (preds == batch.y_who).sum().float() / preds.shape[0]
        return preds, acc, weighted_output

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
    existing_values = worksheet.get_all_values(include_tailing_empty=False)
    start_row = len(existing_values)

    if start_row == 0 or worksheet.get_value('A1') != 'Group':
        worksheet.insert_rows(0, number=1, values=header)
        start_row = 1

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

    worksheet.insert_rows(start_row, number=len(rows), values=rows)


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
        dropout=0.1  # added to avoid missing argument
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

    model_paths = [
        ('/usr/local/checkpoints/GraphLevelNodeLevelMMPN/e1d2c872aa5f48159d16b3d870181b41/epoch=249-val_loss=0.18698-val_acc=0.35377.ckpt','seed3'),
        ('/usr/local/checkpoints/GraphLevelNodeLevelMMPN/52c0ad2a898b4dd99fa2e8168d9e0453/epoch=35-val_loss=0.19694-val_acc=0.29717.ckpt','seed4'),
        ('/usr/local/checkpoints/GraphLevelNodeLevelMMPN/83f9a414a1c24c488d4e4e7cebf317b9/epoch=141-val_loss=0.19594-val_acc=0.27103.ckpt','seed5'),
    ]

    models = []
    val_f1_scores = []

   
    for path, tag in model_paths:
        model = MMPN.NodeLevelMMPN.load_from_checkpoint(path)
        model.to(FLAGS.device)
        model.eval()
        patch_predict_softmax(model)
        models.append(model)

        val_f1 = evaluate_val_f1_for_model(model, val_loader, device=FLAGS.device)
        val_f1_scores.append(val_f1)

    normalized_weights = compute_normalized_f1_weights(models, val_loader, device=FLAGS.device)
    print("Normalized F1 Weights:", normalized_weights.tolist())
    print(f"Ensemble weights: {[round(float(w), 5) for w in normalized_weights]}")

    predict_fn = lambda batch: ensemble_predict(batch, models, normalized_weights)

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






