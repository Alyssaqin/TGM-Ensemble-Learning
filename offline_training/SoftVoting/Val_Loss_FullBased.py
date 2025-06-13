import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from tqdm import tqdm
import os


import MMPN
from dataset import SummerschoolDataset  
from argparse import Namespace
import pygsheets
import config
import utils
from sklearn.metrics import accuracy_score, f1_score

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

def create_weighted_ensemble(model_paths, val_loader, device='cpu'):
    model_paths = sorted(model_paths)
    models = []
    val_losses = []
    loss_fn = nn.MSELoss()

    for path, _ in model_paths:
        model = MMPN.NodeLevelMMPN.load_from_checkpoint(
                checkpoint_path=path,
                n_features_nodes=sum(kept_indices_manual_selection),
                n_features_global=config.NUM_GLOB_FEATURES,
                message_arch=FLAGS.h_mess_arch_1,
                node_arch=FLAGS.h_node_arch_1,
                n_embedding_group=FLAGS.group_embedding_dim,
                n_output_dim_node=1,
                n_output_dim_action=1,
                use_scheduler=True,
                second_layer=FLAGS.GNN_second_layer,
                second_message_arch=FLAGS.h_mess_arch_2,
                second_node_update_arch=FLAGS.h_node_arch_2,
                memory_network_block=nn.LSTM if FLAGS.memory_type == 'LSTM' else nn.GRU,
                lr=FLAGS.learning_rate,
                dropout=0.1,
                teen_group=FLAGS.teen_group,
                feat_sel=feature_sel_dict
            )
        model.to(device)
        model.eval()
        patch_predict_softmax(model)
        val_loss = evaluate_val_loss_for_model(model, val_loader, loss_fn, device)
        models.append(model)
        val_losses.append(val_loss)

    weights = [1.0 / loss if loss > 0 else 0.0 for loss in val_losses]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else [1 / len(weights)] * len(weights)

    print(f"Ensemble weights: {[round(w, 5) for w in normalized_weights]}")
    return models, normalized_weights

def evaluate_val_loss_for_model(model, val_loader, loss_fn, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            x, edge_index = batch.x, batch.edge_index
            global_attr, batch_idx = batch.global_attr, batch.batch
            logits = model.model(x, edge_index, global_attr, 3, 6, batch_idx)
            targets = torch.nn.functional.one_hot(batch.y_who, num_classes=logits.size(1)).float()
            loss = loss_fn(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
    return total_loss / total_graphs if total_graphs > 0 else 0.0

def ensemble_predict(batch, models, normalized_weights):
    with torch.no_grad():
        softmax_outputs = [model.predict_softmax(batch) for model in models]
        weighted_output = torch.zeros_like(softmax_outputs[0])
        for i, output in enumerate(softmax_outputs):
            weighted_output += normalized_weights[i] * output
        preds = weighted_output.argmax(dim=1)
        if hasattr(batch, 'y_who'):
            acc = (preds == batch.y_who).sum().float() / preds.shape[0]
            return preds, acc, weighted_output
        return preds, None, weighted_output

def evaluate_ensemble_f1(predict_fn, dataloader, group_name=""):
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc=f"Evaluating macro-F1 {group_name}"):
        batch = batch.to(FLAGS.device)
        preds, _, _ = predict_fn(batch)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y_who.cpu().tolist())
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"[{group_name}] Ensemble Macro-F1: {round(macro_f1, 4)}")
    return macro_f1
    
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


    model_paths = get_model_paths_from_gsheet(sheet_name="Teenager", sheet_id=2, top_k=3)

    
    models = []
    weights = []
    for path, _ in model_paths:
        model = MMPN.NodeLevelMMPN.load_from_checkpoint(path) 
        model.eval()
        patch_predict_softmax(model)

        loss_fn = nn.MSELoss()
        val_loss = evaluate_val_loss_for_model(model, val_loader, loss_fn, FLAGS.device)

        models.append(model)
        weights.append(1.0 / val_loss if val_loss > 0 else 0.0)

    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights] if total_weight > 0 else [1 / len(weights)] * len(weights)

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

       
        val_f1 = evaluate_ensemble_f1(predict_fn, val_loader_group, group_name=f"{group} val")
        val_acc = evaluate_ensemble(predict_fn, val_loader_group, group_name=f"{group} val")

        test_f1 = evaluate_ensemble_f1(predict_fn, test_loader_group, group_name=f"{group} test")
        test_acc = evaluate_ensemble(predict_fn, test_loader_group, group_name=f"{group} test")

      
        result_dict[group] = {
            "val_f1": val_f1,
            "val_acc": val_acc,
            "test_f1": test_f1,
            "test_acc": test_acc
        }
    write_results_to_google_sheet(
        sheet_name="Result", 
        sheet_id=3,  
        result_dict=result_dict
        )






