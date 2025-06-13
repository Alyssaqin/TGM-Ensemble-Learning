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

import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def get_model_paths_from_gsheet(sheet_name="Teenager", sheet_id=3):
    gc = pygsheets.authorize(service_file="/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json")
    sh = gc.open(sheet_name)
    worksheet = sh[sheet_id]

    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    
    df["Group"] = df["Training Set"].str.extract(r"^(R2D2|WALLE|ICUB)")
    df["val: macro avg:f1"] = pd.to_numeric(df["val: macro avg:f1"], errors='coerce')

    def get_variant_type(tag):
        tag_lower = tag.lower()
        if "lr" in tag_lower:
            return "lr"
        elif "dropout" in tag_lower:
            return "dropout"
        elif any(x in tag_lower for x in ["medium", "shallow", "deep", "default"]):
            return "struct"
        return "other"

    df["variant_type"] = df["Training Set"].apply(get_variant_type)

    selected_models = []
    used_variant_types = set()

    for group in ['ICUB', 'R2D2', 'WALLE']:
        group_df = df[df["Group"] == group].sort_values(by="val: macro avg:f1", ascending=False)

        top_k = group_df.head(5)
        selected = None
        for _, row in top_k.iterrows():
            if row["variant_type"] not in used_variant_types:
                selected = row
                used_variant_types.add(row["variant_type"])
                break
        if selected is None:
            selected = top_k.iloc[0]  # fallback: highest F1, even if same type

        ckpt_path = os.path.abspath(os.path.join(os.getcwd(), selected["Dropout"]))
        selected_models.append((ckpt_path, selected["Training Set"]))

    return selected_models


model_info = get_model_paths_from_gsheet(sheet_name="Teenager", sheet_id=3)
print("Selected top-1 model per group:")
for path, tag in model_info:
    print(f"Tag: {tag}, Path: {path}")

def patch_predict_logits(model):
    def predict_logits(self, batch):
        x, edge_index = batch.x, batch.edge_index
        global_attr, batch_idx = batch.global_attr, batch.batch
        logits = self.model(x, edge_index, global_attr, 3, 6, batch_idx)  
        return logits  
    
    model.predict_logits = types.MethodType(predict_logits, model)


def get_logit_outputs(models, loader, device):
    logit_list = []
    labels = []

    for batch in loader:
        batch = batch.to(device)

        batch_logits = []
        for model in models:
            model.eval()
            with torch.no_grad():
                logits = model.predict_logits(batch)
                batch_logits.append(logits.cpu().numpy())

        logit_list.append(batch_logits)
        labels.append(batch.y_who.cpu().numpy())

    final_logit_list = [np.vstack([batch[i] for batch in logit_list]) for i in range(len(models))]
    labels = np.concatenate(labels)
    return final_logit_list, labels


def evaluate_stacking_on_group(X_test, y_test, meta_model):
    y_pred = meta_model.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    print(f"Stacking Macro-F1: {f1_macro:.4f}")
    print(f"Stacking Accuracy: {acc:.4f}")
    return f1_macro, acc

def evaluate_stacking_on_multiple_groups(models, meta_model, dataset, device, batch_size):
    group_settings = {
        'full': None,
        'ICUB': 'ICUB',
        'R2D2': 'R2D2',
        'WALLE': 'WALLE'
    }

    results = []

    for group_name, fixed_group in group_settings.items():
        print(f"\nEvaluating on group: {group_name}")
        train_ds, val_ds, test_ds, _ = utils.generateTrainDoubleTestValidDataset(
            dataset, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE,
            fixed_group=fixed_group
        )
        _, val_loader, test_loader = utils.loadData(train_ds, val_ds, test_ds, batch_size)

        val_softmax_list, y_val = get_logit_outputs(models, val_loader, device)
        test_softmax_list, y_test = get_logit_outputs(models, test_loader, device)

        X_val = np.hstack(val_softmax_list)
        X_test = np.hstack(test_softmax_list)

        f1_val, acc_val = evaluate_stacking_on_group(X_val, y_val, meta_model)
        f1_test, acc_test = evaluate_stacking_on_group(X_test, y_test, meta_model)

        results.append({
            'group': group_name,
            'f1_val': f1_val,
            'acc_val': acc_val,
            'f1_test': f1_test,
            'acc_test': acc_test
        })

    return results

def write_results_to_google_sheet(sheet_name, sheet_id, all_results_dict):
    import pygsheets
    gc = pygsheets.authorize(service_file='/usr/local/offline_training/apikey/teenagers-450219-36bb8173156a.json')
    sh = gc.open(sheet_name)
    worksheet = sh[sheet_id]

    first_cell = worksheet.get_value('A1')
    if first_cell != 'Meta-Model':
        header = ['Meta-Model', 'Group', 'Val Macro-F1', 'Val Accuracy', 'Test Macro-F1', 'Test Accuracy']
        worksheet.insert_rows(0, number=1, values=header)  
       

    for model_name, result_list in all_results_dict.items():
        for res in result_list:
            row = [
                model_name,
                res['group'],
                round(res['f1_val'], 4),
                round(res['acc_val'], 4),
                round(res['f1_test'], 4),
                round(res['acc_test'], 4)
            ]
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

    model_paths = get_model_paths_from_gsheet(sheet_name="Teenager", sheet_id=3)
    models = []
    for path, _ in model_paths:
        model = MMPN.NodeLevelMMPN.load_from_checkpoint(path) 
        patch_predict_logits(model)
        models.append(model)

    # === Prepare datasets and extract softmax outputs ===
    train_dataset, val_dataset, test_dataset, _ = utils.generateTrainDoubleTestValidDataset(
        dataset, FLAGS.h_SEED, FLAGS.h_TEST_SIZE, FLAGS.h_VALID_SIZE, fixed_group=None
    )
    train_loader, val_loader, test_loader = utils.loadData(train_dataset, val_dataset, test_dataset, FLAGS.h_BATCH_SIZE)

    train_logits, y_train = get_logit_outputs(models, train_loader, FLAGS.device)
    val_logits, y_val = get_logit_outputs(models, val_loader, FLAGS.device)
    test_logits, y_test = get_logit_outputs(models, test_loader, FLAGS.device)
    

    X_train = np.hstack(train_logits)
    X_val = np.hstack(val_logits)
    X_test = np.hstack(test_logits)



    # === Define stacking meta-models ===
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.svm import SVC

    meta_models = {
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3, max_features='sqrt', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True),
        'Logistic Regression': LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000),
    }


    all_results_dict = {}
    for name, meta_model in meta_models.items():
        print(f"\nTraining stacking meta-model: {name}")
        meta_model.fit(X_train, y_train)

        all_results = evaluate_stacking_on_multiple_groups(
            models, meta_model, dataset, FLAGS.device, FLAGS.h_BATCH_SIZE
        )

        all_results_dict[name] = all_results
        for res in all_results:
            print(f"{name} on group {res['group']}: "
                f"Val F1={res['f1_val']:.4f}, Val Acc={res['acc_val']:.4f}, "
                f"Test F1={res['f1_test']:.4f}, Test Acc={res['acc_test']:.4f}")
    
    # === Write to Google Sheet ===
    write_results_to_google_sheet(
        sheet_name='Result',
        sheet_id=4,
        all_results_dict=all_results_dict
    )




    
