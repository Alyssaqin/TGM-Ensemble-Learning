import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
import torch_scatter

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.optim as optim

from torch.optim.lr_scheduler import *

from sklearn.metrics import classification_report,confusion_matrix



class NodeLevelMMPN(pl.LightningModule):

    def __init__(self, n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', loss_module=nn.MSELoss(), split_mode="", lr=0.1, feat_sel=None, memory_network_block=nn.LSTM, teen_group='All'):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.model = MMPN(n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch, 
                    third_layer, third_message_arch, third_node_update_arch, memory_network_block)
        
        
        self.loss_module = loss_module
        self.num_parameters=self.model.num_parameters
        self.optimizer = None

    

    def forward(self, data, mode="train"):
        x, edge_index, global_attr, batch_idx = data.x, data.edge_index, data.global_attr, data.batch
        x = self.model(x, edge_index, global_attr, 3, 6, batch_idx)
      
        preds = x.argmax(dim=1)
       
        loss = self.loss_module(x, torch.nn.functional.one_hot(data.y_who.clone().detach(), num_classes=4).float()) #data.y)
       
        acc = (preds == data.y_who).sum().float() / preds.shape[0]
        return loss, acc, preds, data.y_who

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well 
        print("Call configure")
        optimizer = optim.AdamW(params=self.parameters(),lr=self.learning_rate)
      
        return optimizer
       

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report

    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)

        return dict_report

class BasicLinearHandler(pl.LightningModule):

    def __init__(self, n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes=4, lstm_second=False, loss_module=nn.MSELoss(), split_mode="", lr=0.1, feat_sel=None, memory_network_block=nn.LSTM, teen_group='All'):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.model = SingleVectorNetwork(n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes, lstm_second)
        
        
        self.loss_module = loss_module
        self.num_parameters=self.model.num_parameters
        self.optimizer = None

    

    def forward(self, data, mode="train"):
        x, edge_index, global_attr, batch_idx = data.x, data.edge_index, data.global_attr, data.batch
        x = self.model(x, edge_index, global_attr, 3, 6, batch_idx)
       
        
        preds = x.argmax(dim=1)
       
        loss = self.loss_module(x, torch.nn.functional.one_hot(data.y_who.clone().detach(), num_classes=4).float()) #data.y)

        acc = (preds == data.y_who).sum().float() / preds.shape[0]
        return loss, acc, preds, data.y_who

    def configure_optimizers(self):
        print("Call configure")
        optimizer = optim.AdamW(params=self.parameters(),lr=self.learning_rate)
        
        return optimizer
      

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report

    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)

        return dict_report

class BasicLinearHandlerType(pl.LightningModule):

    def __init__(self, n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes=10, lstm_second=False, loss_module=nn.MSELoss(), split_mode="", lr=0.1, feat_sel=None, memory_network_block=nn.LSTM, teen_group='All'):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.model = SingleVectorNetwork(n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes, lstm_second)
        

        self.loss_module = loss_module
        self.num_parameters=self.model.num_parameters
        self.optimizer = None

    

    def forward(self, data, mode="train"):
        x, edge_index, global_attr, batch_idx = data.x, data.edge_index, data.global_attr, data.batch
        x = self.model(x, edge_index, global_attr, 3, 6, batch_idx)
    
        preds = x.argmax(dim=1)
      
        loss = self.loss_module(x, torch.nn.functional.one_hot(data.y_what.clone().detach(), num_classes=10).float()) #data.y)
        #loss = self.loss_module(x, data.y.float())
        acc = (preds == data.y_what).sum().float() / preds.shape[0]
        return loss, acc, preds, data.y_what

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well 
        print("Call configure")
        optimizer = optim.AdamW(params=self.parameters(),lr=self.learning_rate)
        #optimizer = optim.SGD(self.parameters(), lr=self.learning_rate , momentum=0.9, weight_decay=2e-5)
        # scheduler = ReduceLROnPlateau(optimizer,
        #         #monitor='val_loss',  # Adjust learning rate based on validation loss
        #         factor=0.5,  # Reduce learning rate by half when the monitored metric plateaus
        #         patience=3,  # Number of epochs with no improvement after which learning rate will be reduced
        #         mode='min',  # Monitor for a decrease in the monitored metric
        #         verbose=True  # Print a message when learning rate is adjusted
        #     )
        return optimizer
        # return {
        #             'optimizer': optimizer,
        #             'scheduler': scheduler,
        #             'monitor': 'val_acc'
        #         }

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
        #loss.backward()

    # Access and print gradients
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report

    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)

        return dict_report

class FixedLengthLinearHandler(pl.LightningModule):

    def __init__(self, n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes=2, lstm_second=False, loss_module=nn.MSELoss(), split_mode="", lr=0.1, feat_sel=None, memory_network_block=nn.LSTM, teen_group='All'):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.model = SingleVectorFixedLengthNetwork(n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes, lstm_second)
        
        
        #self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()
        self.loss_module = loss_module
        self.num_parameters=self.model.num_parameters
        self.optimizer = None

    

    def forward(self, data, mode="train"):
        x, edge_index, global_attr, batch_idx = data.x, data.edge_index, data.global_attr, data.batch
        y_out = self.model(x, edge_index, global_attr, 3, 6, batch_idx)
        #x = x.squeeze(dim=-1)
        #print(self.model.group_adr_prediction.weights.grad)
        #if self.hparams.c_out == 1:
        #    preds = (x > 0).float()
        #    data.y = data.y.float()
        #else:
        #preds = x.argmax(dim=-1)
        #print(x, preds)
        # preds = x.argmax(dim=1)
        # #print(preds)
        # #print(x.shape, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).shape)
        # #loss = self.loss_module(x, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).float()) #data.y)
        # loss = self.loss_module(x, torch.nn.functional.one_hot(data.y_who.clone().detach(), num_classes=4).float()) #data.y)
        # #loss = self.loss_module(x, data.y.float())
        # acc = (preds == data.y_who).sum().float() / preds.shape[0]
        loss = self.loss_module(y_out,data.y_who_one.clone().detach().float()) #data.y)
        preds_local = y_out.argmax(dim=1)
        #print(y_out)
        #print(preds_local.shape, preds_local)
        has_selected, _ = torch_scatter.scatter_min(preds_local, batch_idx, dim=0)
        #print(has_selected)
        has_selected = has_selected==1
        #print(has_selected)
        #print(has_selected)
        #print(data.y_who_one)
        #loss = self.loss_module(y_out,data.y_who_local.clone().detach())
        
        # preds = y_out.argmax(dim=-1)
        # acc = (preds.clone().detach() == data.y_who).sum().float() / preds.shape[0]
        #print(y_out[:,0].shape)
        preds_val, preds_batch_id = torch_scatter.scatter_max(y_out[:,0], batch_idx, dim=0)
        #     #print(preds_val.shape)
        target_ids_batches = []
        for n in range(global_attr.shape[0]):
            target_ids_batches.extend(np.arange(3))
        
        index_preds_batch = np.array(preds_batch_id.squeeze()).astype(int)
        preds = torch.tensor(np.array(target_ids_batches)[index_preds_batch])
   
        # if scatter_min results in voerall 1, then no person was selected and we select the group insteads
        preds[has_selected] == 3

        #     target_vec = []
        #     for line in data.y_who_one:
        #         #print(line)
        #         target_vec.extend(line)
        #     #print(data.y_who_one)
        #     #print(y_out.squeeze().shape, torch.tensor(target_vec).float().shape)
        #     loss = self.loss_module(y_out.squeeze(), torch.tensor(target_vec).float()) #data.y)
        
        #acc = (preds == data.y_who).sum().float() / preds.shape[0]
        
        acc = (preds == data.y_who).sum().float() / preds.shape[0]
 
        return loss.float(), acc, preds.clone().detach(), data.y_who

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well 
        print("Call configure")
        optimizer = optim.AdamW(params=self.parameters(),lr=self.learning_rate)
        #optimizer = optim.SGD(self.parameters(), lr=self.learning_rate , momentum=0.9, weight_decay=2e-5)
        # scheduler = ReduceLROnPlateau(optimizer,
        #         #monitor='val_loss',  # Adjust learning rate based on validation loss
        #         factor=0.5,  # Reduce learning rate by half when the monitored metric plateaus
        #         patience=3,  # Number of epochs with no improvement after which learning rate will be reduced
        #         mode='min',  # Monitor for a decrease in the monitored metric
        #         verbose=True  # Print a message when learning rate is adjusted
        #     )
        return optimizer
        # return {
        #             'optimizer': optimizer,
        #             'scheduler': scheduler,
        #             'monitor': 'val_acc'
        #         }

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
        #loss.backward()

        # Access and print gradients
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report

    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)

        return dict_report

class FixedLengthLinearHandlerType(pl.LightningModule):

    def __init__(self, n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes=10, lstm_second=False, loss_module=nn.MSELoss(), split_mode="", lr=0.1, feat_sel=None, memory_network_block=nn.LSTM, teen_group='All'):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.model = SingleVectorFixedLengthNetworkType(n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes, lstm_second)
        
        
        #self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()
        self.loss_module = loss_module
        self.num_parameters=self.model.num_parameters
        self.optimizer = None

    

    def forward(self, data, mode="train"):
        x, edge_index, global_attr, batch_idx = data.x, data.edge_index, data.global_attr, data.batch
        y_out = self.model(x, edge_index, global_attr, 3, 6, batch_idx)
        #x = x.squeeze(dim=-1)
        #print(self.model.group_adr_prediction.weights.grad)
        #if self.hparams.c_out == 1:
        #    preds = (x > 0).float()
        #    data.y = data.y.float()
        #else:
        #preds = x.argmax(dim=-1)
        #print(x, preds)
        # preds = x.argmax(dim=1)
        # #print(preds)
        # #print(x.shape, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).shape)
        # #loss = self.loss_module(x, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).float()) #data.y)
        # loss = self.loss_module(x, torch.nn.functional.one_hot(data.y_who.clone().detach(), num_classes=4).float()) #data.y)
        # #loss = self.loss_module(x, data.y.float())
        # acc = (preds == data.y_who).sum().float() / preds.shape[0]
        preds = y_out.argmax(dim=1)
        #print(preds)
        #print(x.shape, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).shape)
        #loss = self.loss_module(x, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).float()) #data.y)
        loss = self.loss_module(y_out, torch.nn.functional.one_hot(data.y_what.clone().detach(), num_classes=10).float()) #data.y)
        #loss = self.loss_module(x, data.y.float())
        acc = (preds == data.y_what).sum().float() / preds.shape[0]
        return loss, acc, preds, data.y_what

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well 
        print("Call configure")
        optimizer = optim.AdamW(params=self.parameters(),lr=self.learning_rate)
        #optimizer = optim.SGD(self.parameters(), lr=self.learning_rate , momentum=0.9, weight_decay=2e-5)
        # scheduler = ReduceLROnPlateau(optimizer,
        #         #monitor='val_loss',  # Adjust learning rate based on validation loss
        #         factor=0.5,  # Reduce learning rate by half when the monitored metric plateaus
        #         patience=3,  # Number of epochs with no improvement after which learning rate will be reduced
        #         mode='min',  # Monitor for a decrease in the monitored metric
        #         verbose=True  # Print a message when learning rate is adjusted
        #     )
        return optimizer
        # return {
        #             'optimizer': optimizer,
        #             'scheduler': scheduler,
        #             'monitor': 'val_acc'
        #         }

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
        #loss.backward()

        # Access and print gradients
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report

    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)

        return dict_report

class HandleMMPNType(NodeLevelMMPN):
    def __init__(self, n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', loss_module=nn.MSELoss(), split_mode="", lr=0.1, feat_sel=None, memory_network_block=nn.LSTM, teen_group='All'):
        super().__init__(n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch, 
                    third_layer, third_message_arch, third_node_update_arch, loss_module, split_mode, lr, feat_sel, memory_network_block)
        self.model = MMPNType(n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch, 
                    third_layer, third_message_arch, third_node_update_arch, memory_network_block)

    def forward(self, data, mode="train"):
        #print("Forward Type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        x, edge_index, global_attr, batch_idx, chosen_who = data.x, data.edge_index, data.global_attr, data.batch, data.y_who
        #print(chosen_who)
        x = self.model(x, edge_index, global_attr, 3, 6, batch_idx, chosen_who)
        #x = x.squeeze(dim=-1)
        
        #if self.hparams.c_out == 1:
        #    preds = (x > 0).float()
        #    data.y = data.y.float()
        #else:
        preds = x.argmax(dim=1)
        #print(x.shape, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).shape)
        #loss = self.loss_module(x, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).float()) #data.y)
        loss = self.loss_module(x, torch.nn.functional.one_hot(data.y_what.clone().detach(), num_classes=10).float()) #data.y)
        #loss = self.loss_module(x, data.y.float())
        acc = (preds == data.y_what).sum().float() / preds.shape[0]
        return loss, acc, preds, data.y_what
    
class HandleMMPNTiming(NodeLevelMMPN):
    def __init__(self, n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', loss_module=nn.MSELoss(), split_mode="", lr=0.1, feat_sel=None, memory_network_block=nn.LSTM):
        super().__init__(n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch, 
                    third_layer, third_message_arch, third_node_update_arch, loss_module, split_mode, lr, feat_sel, memory_network_block)
        self.model = MMPNTiming(n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch, 
                    third_layer, third_message_arch, third_node_update_arch, memory_network_block)
        self.num_parameters = self.model.num_parameters

    def forward(self, data, mode="train"):
        #print("Forward Type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        x, edge_index, global_attr, batch_idx = data.x, data.edge_index, data.global_attr, data.batch
        x = self.model(x, edge_index, global_attr, 3, 6, batch_idx)
        #x = x.squeeze(dim=-1)
        
        #if self.hparams.c_out == 1:
        #    preds = (x > 0).float()
        #    data.y = data.y.float()
        #else:
        preds = x.argmax(dim=-1)
        #print(x.shape, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).shape)
        #loss = self.loss_module(x, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).float()) #data.y)
        loss = self.loss_module(x, torch.nn.functional.one_hot(data.y_timing.clone().detach(), num_classes=2).float()) #data.y)
        #loss = self.loss_module(x, data.y.float())
        acc = (preds == data.y_timing).sum().float() / preds.shape[0]
        return loss, acc, preds, data.y_timing

class NodeLevelMMPNTimeFree(pl.LightningModule):

    def __init__(self, n_features_nodes,n_features_edge, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='',dropout_p=0.5, pooling_operation='max',
                    third_layer=False, third_message_arch='', third_node_update_arch='', loss_module=nn.MSELoss(), split_mode="", lr=0.1, limit_train=False, feat_sel=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.model = MMPNTimeFree(n_features_nodes, n_features_edge, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer, second_message_arch, second_node_update_arch, 
                    third_layer, third_message_arch, third_node_update_arch, pooling_operation, dropout_p)
        
        
        #self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()
        self.loss_module = loss_module
        self.num_parameters = self.model.num_parameters

    

    def forward(self, data, mode="train"):
        x, edge_index, global_attr, batch_idx = data.x, data.edge_index, \
                                                            data.global_attr, \
                                                                data.batch
        num_nodes = torch.Tensor(np.array(data.num_nodes_loc)).to(torch.int64).to(device=global_attr.device).squeeze(1)
        num_edges = torch.Tensor(np.array(data.num_edges_loc)).to(torch.int64).to(device=global_attr.device).squeeze(1)
        #print(num_edges, global_attr.shape, x.shape)
        #print('Before forward')
        x = self.model(x, edge_index, global_attr, num_nodes, num_edges, batch_idx)
        #print('After forward')
        preds = x.argmax(dim=1)
        #print(preds)
        #print(x.shape, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).shape)
        #loss = self.loss_module(x, torch.nn.functional.one_hot(torch.tensor(data.y), num_classes=4).float()) #data.y)
        
        # MSELOSS
        loss = self.loss_module(x, torch.nn.functional.one_hot(data.y_who.clone().detach(), num_classes=4).float())
        
        
        #loss = self.loss_module(x, data.y_who).float()
        #print(loss, acc)
        acc = (preds == data.y_who).sum().float() / preds.shape[0]
        return loss.float(), acc, preds, data.y_who

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well 
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate , momentum=0.9, weight_decay=2e-5)
        #optimizer = torch.optim.Adam(self.parameters(), lr=10e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch, mode="train")
        #loss = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="val")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('val-'+key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('val-'+key, dict_report[key])
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return dict_report
    
    def test_step(self, batch, batch_idx):
        loss, acc, pred, y = self.forward(batch, mode="test")
        dict_report = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True, zero_division=0)
        for key in dict_report:
            if type(dict_report[key]) == dict:
                for key_small in dict_report[key]:
                    self.log('test-'+ key+'-'+key_small, dict_report[key][key_small])
            else:
                self.log('test-'+key, dict_report[key])
        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return dict_report

class MMPNTimeFree(nn.Module):

    def __init__(self, n_features_nodes, n_features_edges, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', pooling_operation='max',dropout_p=0.5):
        super(MMPNTimeFree, self).__init__()
        print('Used dropout:', dropout_p)
        self.pooling_operation = pooling_operation
        self.second_layer = second_layer
        self.third_layer = third_layer
        self.message_nn = torch.nn.Sequential()
        self.edge_features = n_features_edges
        layers = message_arch.split('-')
        n_message_dim = int(layers[-1])
        input_dim = n_features_edges+n_features_nodes+n_features_nodes+n_features_global
        for i in range(len(layers)):
            self.message_nn.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.message_nn.add_module("Relu"+str(i), nn.ReLU())
            if i != len(layers) - 1:
                self.message_nn.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
            input_dim = int(layers[i])
        #new_edge_dim = input_dim
        # self.message_nn = nn.Sequential(
        #     nn.Linear(n_features_edges+n_features_nodes+n_features_nodes+n_features_global, n_message_dim_mid_layer),
        #     nn.ReLU(),
        #     nn.Linear(n_message_dim_mid_layer, n_message_dim)
        # )
        print('Input dim message', n_features_edges+n_features_nodes+n_features_nodes+n_features_global)
        #nn.Linear(n_features_edges+n_features_nodes+n_features_nodes, n_message_dim)

        #self.update_nn = nn.Linear(n_message_dim + n_features_nodes, n_embedding_dim)
        self.update_nn = torch.nn.Sequential()
        layers = node_arch.split('-')
        input_dim = n_message_dim + n_features_nodes + n_features_global
        n_embedding_dim = int(layers[-1])
        print(layers, n_embedding_dim)
        for i in range(len(layers)):
            self.update_nn.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.update_nn.add_module("Relu"+str(i), nn.ReLU())
            if i != len(layers) - 1:
                self.update_nn.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
            input_dim = int(layers[i])
        #new_node_dim = input_dim

        # update global
        self.to_global_nn = nn.Linear(n_embedding_dim+n_features_global, n_embedding_group)

        print('Input dim update: ', n_message_dim + n_features_nodes + n_features_global)


        if second_layer:
            # second round of GNN
            #self.message_nn_2 = = torch.nn.Sequential() 
            #nn.Linear(n_message_dim+n_embedding_dim+n_embedding_dim, n_message_dim)
            self.message_nn_2 = torch.nn.Sequential()
            layers = second_message_arch.split('-')
            n_message_dim_2 = int(layers[-1])
            print(layers, n_message_dim_2)
            print("Second message dim", n_message_dim_2)
            input_dim = n_message_dim+n_embedding_dim+n_embedding_dim+n_embedding_group
            for i in range(len(layers)):
                self.message_nn_2.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
                self.message_nn_2.add_module("Relu"+str(i), nn.ReLU())
                if i != len(layers) - 1:
                    self.message_nn_2.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
                input_dim = int(layers[i])


            self.update_nn_2 = torch.nn.Sequential()
            layers = second_node_update_arch.split('-')
            
            print('Input to second update', n_message_dim_2, n_embedding_dim, n_embedding_group)
            input_dim = n_message_dim_2+n_embedding_dim + n_embedding_group
            n_embedding_dim = int(layers[-1])
            for i in range(len(layers)):
                self.update_nn_2.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
                self.update_nn_2.add_module("Relu"+str(i), nn.ReLU())
                if i != len(layers) - 1:
                    self.update_nn_2.add_module("Dropout"+str(i), nn.Dropout(dropout_p))
                input_dim = int(layers[i])
            
            self.to_global_nn_2 = nn.Linear(n_embedding_dim+n_embedding_group, n_embedding_group)

        if third_layer:
            # second round of GNN
            #self.message_nn_2 = = torch.nn.Sequential() 
            #nn.Linear(n_message_dim+n_embedding_dim+n_embedding_dim, n_message_dim)
            self.message_nn_3 = torch.nn.Sequential()
            layers = third_message_arch.split('-')
            n_message_dim_3 = int(layers[-1])
            print(layers, n_message_dim_3)
            print("Second message dim", n_message_dim_3)
            input_dim = n_message_dim_2+n_embedding_dim+n_embedding_dim+n_embedding_group
            for i in range(len(layers)):
                self.message_nn_3.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
                self.message_nn_3.add_module("Relu"+str(i), nn.ReLU())
                input_dim = int(layers[i])


            self.update_nn_3 = torch.nn.Sequential()
            layers = third_node_update_arch.split('-')
            
            print('Input to second update', n_message_dim_3, n_embedding_dim, n_embedding_group)
            input_dim = n_message_dim_3+n_embedding_dim + n_embedding_group
            n_embedding_dim = int(layers[-1])
            for i in range(len(layers)):
                self.update_nn_3.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
                self.update_nn_3.add_module("Relu"+str(i), nn.ReLU())
                input_dim = int(layers[i])

        # third round of GNN predict
        self.embedding_nn = nn.Linear(n_embedding_dim,n_output_dim_node)
  

        self.last_relu = nn.ReLU()
        self.group_adr_prediction = nn.Linear(n_embedding_group, n_output_dim_action)
        self.softmax = nn.Softmax(dim=1)
        print('Size input = ', n_embedding_dim, '+', n_embedding_group)
        # the input features consist only of the updated output of one node -the chosen node-, 
        # alternatively it could get all nodes but with an additional feature indicating that this node was chosen
        self.num_parameters = self.count_parameters(self )
        self.print_parameters()
    
    def print_parameters(self):
        print('Message model:', self.count_parameters(self.message_nn ), self.message_nn)
        print('Update model:', self.count_parameters(self.update_nn ), self.update_nn)
        if self.second_layer:
            print('Message model 2:', self.count_parameters(self.message_nn_2 ), self.message_nn_2)
            print('Update model 2:', self.count_parameters(self.update_nn_2 ), self.update_nn_2)
        print('Last layer', self.count_parameters(self.embedding_nn ), self.embedding_nn)
        print('Overall:', self.count_parameters(self ))

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    ## TODO: change name once done
    def forward(self, nodes, edge_indices, global_attr, num_nodes, num_edges, batch_indices, return_embeddings=False, edge_attr = None):

        # dims: batch_size, num_nodes, features
        edges_list_index_to_list = []
        overall_count = 0
        #print(edge_indices)
        #print(num_edges)
        #print(nodes.shape)
        #print( edge_attr.shape)
        #print(global_attr.shape)
        #print('Just starting')
        # print(edge_indices)
        # for elem in edge_indices:
        #     print(elem)
        #     count = overall_count
        #     max_elem = 0
        #     list_to_append = []
        #     for edge in elem:
        #         max_elem = max(max_elem, max(edge))
        #         list_to_append.append([x+count for x in edge])
            
        #     if overall_count+max_elem < nodes.shape[0]:
        #         edges_list_index_to_list.extend(list_to_append)     
        #     overall_count += max_elem+1
        #print('Past loop')
        src_index = edge_indices[0].clone().detach() #[x[0] for x in edges_list_index_to_list], device=nodes.device).to(torch.int64)
        target_index = edge_indices[1].clone().detach() #[x[1] for x in edges_list_index_to_list], device=nodes.device).to(torch.int64)
        # dims for edges
        x_nodes = nodes
        #print(len(edges_list_index_to_list))
        tmp_src = torch.index_select(x_nodes, 0, src_index) # for a in x_nodes ])
        
        tmp_target = torch.index_select(x_nodes, 0, target_index) #for a in x_nodes ])
        #print('421')
        # repeat global graph attribute to be part of each edge
        #num_edge_torch = torch.Tensor(np.array(num_edges)).to(torch.int64).to(device=global_attr.device).squeeze(1)
        #print(num_edges.shape, global_attr.shape)
        tmp_glob = torch.repeat_interleave(global_attr, num_edges, 0)
        #print(num_edges, src_index.shape, target_index.shape, tmp_src.shape, edge_attr.shape, tmp_glob.shape, tmp_target.shape)
        #print(num_edges, tmp_src.shape, edge_attr.shape, tmp_target.shape, tmp_glob.shape)
        #print(np.sum(np.array(num_edges)))
        # concatenate node and edge information
        if self.edge_features == 0:
            tmp_concat = torch.cat([tmp_src, tmp_target, tmp_glob], 1)
        else:
            tmp_concat = torch.cat([tmp_src, edge_attr, tmp_target, tmp_glob], 1)
        
        # first pass the message
        # input: node_source + each outgoing edge
        # reduce edge dimension
        message = F.relu(self.message_nn(tmp_concat))
        
        # resort to edges to batches
        #message = message_tmp #.view(-1, len(src_index), message_tmp.shape[1])
        
        #expand target index to be applicable for every dimension of the embedding
        
        target_index_expand = target_index.expand((message.shape[1], target_index.size(0))).T
        
        # output: message_dim for each edge+node combination
        # aggregate information through max (most important information) for the message-incoming nodes
        # note: this code cannot deal with nodes without incoming edges
        # was scatter_max
        #print(message.shape, target_index_expand.shape)
        #output_tensor_aggr_tmp, _ =  torch_scatter.scatter_min(message, target_index_expand, dim=0) #for a in torch.Tensor(message) ])
        if self.pooling_operation == 'mean':
            output_tensor_aggr =  torch_scatter.scatter_mean(message, target_index_expand, dim=0) #for a in torch.Tensor(message) ])
        elif self.pooling_operation == 'min':
            output_tensor_aggr, _ =  torch_scatter.scatter_min(message, target_index_expand, dim=0)
        else:
            output_tensor_aggr, _ =  torch_scatter.scatter_max(message, target_index_expand, dim=0)
        #print(output_tensor_aggr.shape)
        #output_tensor_aggr = output_tensor_aggr_tmp #.view(-1,x_nodes.shape[1],message.shape[2])
        #num_nodes_torch = torch.Tensor(np.array(num_nodes)).to(torch.int64).to(device=global_attr.device).squeeze(1)
        tmp_glob_for_nodes = torch.repeat_interleave(global_attr, num_nodes, 0)
        #print(x_nodes.shape, tmp_glob_for_nodes.shape)
        # use aggregated message information and node features to update the node
        tmp_concat_node_update = torch.cat([x_nodes, output_tensor_aggr, tmp_glob_for_nodes], 1)
        #print('Node update input', tmp_concat_node_update.shape)
        updated_node_embedding = self.update_nn(tmp_concat_node_update) #.view(-1, tmp_concat_node_update.shape[2]))
        # use embedding to aggregate information for global graph attribute
        #print(updated_node_embedding.shape, batch_indices.shape)
        #aggregate_nodes, _ = torch_scatter.scatter_min(updated_node_embedding, batch_indices, dim=0)
        if self.pooling_operation == 'mean':
            aggregate_nodes = torch_scatter.scatter_mean(updated_node_embedding, batch_indices, dim=0)
        elif self.pooling_operation == 'min':
            aggregate_nodes, _ = torch_scatter.scatter_min(updated_node_embedding, batch_indices, dim=0)
        else:
            aggregate_nodes, _ = torch_scatter.scatter_max(updated_node_embedding, batch_indices, dim=0)
        # ToDo think about and check dimensions!
        #tmp_glob_for_nodes = torch.repeat_interleave(global_attr, aggregate_nodes.shape[0], 0)
        #print(aggregate_nodes.shape, global_attr.shape)
        # use global graph attribute to learn group embedding
        #print(aggregate_nodes)
        #print(aggregate_nodes.shape)
        #print(global_attr.shape)
        group_embedding = F.relu(self.to_global_nn(torch.cat([aggregate_nodes, global_attr], 1)))
        #print('485')
        if self.second_layer:
            tmp_src_2 = torch.index_select(updated_node_embedding, 0, src_index) # for a in x_nodes ])
            
            tmp_target_2 = torch.index_select(updated_node_embedding, 0, target_index) #for a in x_nodes ])


            tmp_glob_2 = torch.repeat_interleave(group_embedding, num_edges, 0)

            #print('Shapes for message:', tmp_src_2.shape, message.shape, tmp_target_2.shape, tmp_glob_2.shape)
            tmp_concat_2 = torch.cat([tmp_src_2, message, tmp_target_2, tmp_glob_2], 1)
            

            # first pass the message
            # input: node_source + each outgoing edge
            # reduce edge dimension
            #print(tmp_concat_2.shape)
            message_tmp_2 = self.message_nn_2(tmp_concat_2)
            
            # resort to edges to batches
            message_2 = message_tmp_2 #.view(-1, len(src_index), message_tmp.shape[1])
            
            #expand target index to be applicable for every dimension of the embedding
            
            target_index_expand_2 = target_index.expand((message_2.shape[1], target_index.size(0))).T
            
            # output: message_dim for each edge+node combination
            # aggregate information through max (most important information) for the message-incoming nodes
            # note: this code cannot deal with nodes without incoming edges
            # was scatter_max
            if self.pooling_operation == 'mean':
                output_tensor_aggr_tmp_2 =  torch_scatter.scatter_mean(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            elif self.pooling_operation == 'min':
                output_tensor_aggr_tmp_2, _ =  torch_scatter.scatter_min(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            else:
                output_tensor_aggr_tmp_2, _ =  torch_scatter.scatter_max(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            #output_tensor_aggr_tmp_2 =  torch_scatter.scatter_mean(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            
            output_tensor_aggr_2 = output_tensor_aggr_tmp_2 #.view(-1,x_nodes.shape[1],message.shape[2])
            #num_nodes_torch = torch.Tensor(np.array(num_nodes)).to(torch.int64).to(device=global_attr.device).squeeze(1)
            tmp_glob_for_nodes_2 = torch.repeat_interleave(group_embedding, num_nodes, 0)
            #print(x_nodes.shape, tmp_glob_for_nodes.shape)

            #print("Before concat", updated_node_embedding.shape, output_tensor_aggr_2.shape, tmp_glob_for_nodes_2.shape)    
            # use aggregated message information and node features to update the node
            tmp_concat_node_update_2 = torch.cat([updated_node_embedding, output_tensor_aggr_2, tmp_glob_for_nodes_2], 1)
            #print(tmp_concat_node_update_2.shape)
            #print('Node update input', tmp_concat_node_update.shape)
            updated_node_embedding = self.update_nn_2(tmp_concat_node_update_2)
            if self.pooling_operation == 'mean':
                aggregate_nodes = torch_scatter.scatter_mean(updated_node_embedding, batch_indices, dim=0)
                #output_tensor_aggr_tmp_2 =  torch_scatter.scatter_mean(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            elif self.pooling_operation == 'min':
                aggregate_nodes, _ = torch_scatter.scatter_min(updated_node_embedding, batch_indices, dim=0)
                #output_tensor_aggr_tmp_2, _ =  torch_scatter.scatter_min(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            else:
                aggregate_nodes, _ = torch_scatter.scatter_max(updated_node_embedding, batch_indices, dim=0)
            
            
            group_embedding = self.to_global_nn_2(torch.cat([aggregate_nodes, group_embedding], 1))

            if self.third_layer:
                raise NotImplementedError
                #aggregate_nodes = torch_scatter.scatter_mean(updated_node_embedding, batch_indices, dim=0)
                group_embedding = F.relu(group_embedding_2) #self.to_global_nn_2(torch.cat([aggregate_nodes, group_embedding], 1)))

                tmp_src_3 = torch.index_select(updated_node_embedding, 0, src_index) # for a in x_nodes ])
                
                tmp_target_3 = torch.index_select(updated_node_embedding, 0, target_index) #for a in x_nodes ])


                tmp_glob_3 = torch.repeat_interleave(group_embedding_2, num_edges, 0)

                #print('Shapes for message:', tmp_src_2.shape, message.shape, tmp_target_2.shape, tmp_glob_2.shape)
                tmp_concat_3 = torch.cat([tmp_src_3, message_2, tmp_target_3, tmp_glob_3], 1)
                

                # first pass the message
                # input: node_source + each outgoing edge
                # reduce edge dimension
                #print(tmp_concat_2.shape)
                message_tmp_3 = self.message_nn_3(tmp_concat_3)
                
                # resort to edges to batches
                message_3 = message_tmp_3 #.view(-1, len(src_index), message_tmp.shape[1])
                
                #expand target index to be applicable for every dimension of the embedding
                
                target_index_expand_3 = target_index.expand((message_3.shape[1], target_index.size(0))).T
                
                # output: message_dim for each edge+node combination
                # aggregate information through max (most important information) for the message-incoming nodes
                # note: this code cannot deal with nodes without incoming edges
                output_tensor_aggr_tmp_3, _ =  torch_scatter.scatter_min(message_3, target_index_expand_3, dim=0) #for a in torch.Tensor(message) ])
                
                output_tensor_aggr_3 = output_tensor_aggr_tmp_3 #.view(-1,x_nodes.shape[1],message.shape[2])
                #num_nodes_torch = torch.Tensor(np.array(num_nodes)).to(torch.int64).to(device=global_attr.device).squeeze(1)
                tmp_glob_for_nodes_3 = torch.repeat_interleave(group_embedding_2, num_nodes, 0)
                #print(x_nodes.shape, tmp_glob_for_nodes.shape)

                #print("Before concat", updated_node_embedding.shape, output_tensor_aggr_2.shape, tmp_glob_for_nodes_2.shape)    
                # use aggregated message information and node features to update the node
                tmp_concat_node_update_3 = torch.cat([updated_node_embedding, output_tensor_aggr_3, tmp_glob_for_nodes_3], 1)
                #print(tmp_concat_node_update_2.shape)
                #print('Node update input', tmp_concat_node_update.shape)
                updated_node_embedding = self.update_nn_3(tmp_concat_node_update_3)


        #print(updated_node_embedding.shape)
        # use embedding to get final Q value for each node
        result_per_node = self.embedding_nn(updated_node_embedding)
        #print(result_per_node.shape)
        result_group = self.group_adr_prediction(group_embedding)
        #result_per_node = torch.squeeze(results_per_node[:,-1,:], dim=1)
        #result_per_node = result_per_node.view(-1, num_nodes)
        result_per_node = result_per_node.view(-1, 3)
        #print(result_per_node.shape)
        #results_group = results_group[:,-1,:]
        #print(result_per_node.shape, results_actions.shape)
        result = torch.cat([result_per_node, result_group], 1)
        #chosen_nodes = torch.index_select(updated_node_embedding, 0, actions_batch_wide)
        return self.softmax(result)
        #tmp_concat_action_pred = torch.cat([chosen_nodes, group_embedding], 1)

        #print(result_per_node.shape)
        #results_actions = F.relu(self.action_prediction(tmp_concat_action_pred))
        #print(result_per_node.T)
        #print(batch_indices)
        #print(torch_scatter.scatter_softmax(result_per_node, batch_indices, dim=0).T)
        # if return_embeddings:
        #     return torch_scatter.scatter_softmax(result_per_node, batch_indices, dim=0), updated_node_embedding
        # else:
        #     return torch_scatter.scatter_softmax(result_per_node, batch_indices, dim=0) #, results_actions #.view(-1, x_nodes.shape[1])
 
 
class MMPN(nn.Module):

    def __init__(self, n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', memory_network_block=nn.LSTM, dropout=0.2, num_mem_layers=1):
        super(MMPN, self).__init__()
        self.gamma = 2.0
        self.alpha = 0.25
        n_features_edges = 0
        self.second_layer = second_layer
        self.third_layer = third_layer
        # Sequential to add multiple modules, this does not mean LINEAR layer!
        self.message_nn = torch.nn.Sequential()
        layers = message_arch.split('-')
        n_message_dim = int(layers[-1])
        input_dim = n_features_edges+n_features_nodes+n_features_nodes+n_features_global
        for i in range(len(layers)):
            self.message_nn.add_module("LSTM"+str(i), memory_network_block(input_dim, int(layers[i]), num_layers=num_mem_layers, dropout=dropout))
            #self.message_nn.add_module("Relu"+str(i), nn.ReLU())
            input_dim = int(layers[i])
        #print('Droupout_rate', dropout)
        print('Input dim message', n_features_edges+n_features_nodes+n_features_nodes+n_features_global)
        
        self.update_nn = torch.nn.Sequential()
        layers = node_arch.split('-')
        input_dim = n_message_dim + n_features_nodes + n_features_global
        n_embedding_dim = int(layers[-1])
        print(layers, n_embedding_dim)
        for i in range(len(layers)):
            self.update_nn.add_module("LSTM"+str(i), memory_network_block(input_dim, int(layers[i]), num_layers=num_mem_layers,dropout=dropout))
            #self.update_nn.add_module("Relu"+str(i), nn.ReLU())
            input_dim = int(layers[i])

        # update global
        self.to_global_nn = memory_network_block(n_embedding_dim+n_features_global, n_embedding_group, num_layers=num_mem_layers,dropout=dropout)

        print('Input dim update: ', n_message_dim + n_features_nodes + n_features_global)


        if second_layer:
            # second round of GNN
            self.message_nn_2 = torch.nn.Sequential()
            layers = second_message_arch.split('-')
            n_message_dim_2 = int(layers[-1])
            print(layers, n_message_dim_2)
            print("Second message dim", n_message_dim_2)
            input_dim = n_message_dim+n_embedding_dim+n_embedding_dim+n_embedding_group
            for i in range(len(layers)):
                self.message_nn_2.add_module("LSTM"+str(i), memory_network_block(input_dim, int(layers[i]),num_layers=num_mem_layers,dropout=dropout))
                #self.message_nn_2.add_module("Relu"+str(i), nn.ReLU())
                input_dim = int(layers[i])


            self.update_nn_2 = torch.nn.Sequential()
            layers = second_node_update_arch.split('-')
            
            print('Input to second update', n_message_dim_2, n_embedding_dim, n_embedding_group)
            input_dim = n_message_dim_2+n_embedding_dim + n_embedding_group
            n_embedding_dim = int(layers[-1])
            for i in range(len(layers)):
                self.update_nn_2.add_module("LSTM"+str(i), memory_network_block(input_dim, int(layers[i]), num_layers=num_mem_layers, dropout=dropout))
                #self.update_nn_2.add_module("Relu"+str(i), nn.ReLU())
                input_dim = int(layers[i])
            
            self.to_global_nn_2 = memory_network_block(n_embedding_dim+n_embedding_group, n_embedding_group, num_layers=num_mem_layers, dropout=dropout)

        # third round of GNN predict
        self.embedding_nn = memory_network_block(n_embedding_dim,n_output_dim_node, num_layers=num_mem_layers, dropout=dropout)

        self.group_adr_prediction = memory_network_block(n_embedding_group, n_output_dim_action, num_layers=num_mem_layers, dropout=dropout)
        self.softmax = nn.Softmax(dim=1)
       
        print('Size input = ', n_embedding_dim, '+', n_embedding_group)
        # the input features consist only of the updated output of one node -the chosen node-, 
        # alternatively it could get all nodes but with an additional feature indicating that this node was chosen
        
        self.print_parameters()
        self.num_parameters = self.count_parameters(self )
    
    def print_parameters(self):
        print('Message model:', self.count_parameters(self.message_nn ), self.message_nn)
        print('Update model:', self.count_parameters(self.update_nn ), self.update_nn)
        print('Global model:', self.count_parameters(self.to_global_nn ), self.to_global_nn)
        if self.second_layer:
            print('Message model 2:', self.count_parameters(self.message_nn_2 ), self.message_nn_2)
            print('Update model 2:', self.count_parameters(self.update_nn_2 ), self.update_nn_2)
            print('Global model 2:', self.count_parameters(self.to_global_nn_2 ), self.to_global_nn_2)
        print('Last layer', self.count_parameters(self.embedding_nn ), self.embedding_nn)
        print('Group prediction', self.count_parameters(self.group_adr_prediction ), self.group_adr_prediction)
        print('Overall:', self.count_parameters(self ))

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    ## TODO: change name once done
    def forward(self, nodes, edge_indices, global_attr, num_nodes, num_edges, batch_indices, return_embeddings=False):

        # dims: batch_size, num_nodes, features
        edges_list_index_to_list = []
        overall_count = 0
     
        
        src_index = edge_indices[0] 
        target_index = edge_indices[0] 
        x_nodes = nodes
        
        tmp_src = torch.index_select(x_nodes, 0, src_index) # for a in x_nodes ])
        
        tmp_target = torch.index_select(x_nodes, 0, target_index) #for a in x_nodes ])
        
        # repeat global graph attribute to be part of each edge

        global_attr = global_attr.view(-1, x_nodes.shape[1], global_attr.shape[1])

        tmp_glob = torch.repeat_interleave(global_attr, num_edges, 0)
       
        # concatenate node and edge information

        tmp_concat = torch.cat([tmp_src, tmp_target, tmp_glob], 2)
        
        
        message_result, _ = self.message_nn(tmp_concat)

        message_tmp = F.relu(message_result)
        
        # resort to edges to batches
        message = message_tmp #.view(-1, len(src_index), message_tmp.shape[1])
        
        #expand target index to be applicable for every dimension of the embedding
        
        target_index_expand = target_index.expand((message.shape[1], target_index.size(0))).T
        
      
        output_tensor_aggr_tmp, _ =  torch_scatter.scatter_max(message, target_index_expand, dim=0) #for a in torch.Tensor(message) ])
       
        output_tensor_aggr = output_tensor_aggr_tmp 
      
        tmp_glob_for_nodes = torch.repeat_interleave(global_attr, num_nodes, 0)
      
        tmp_concat_node_update = torch.cat([x_nodes, output_tensor_aggr, tmp_glob_for_nodes], 2)
      
        updated_node_embedding, _ = self.update_nn(tmp_concat_node_update) #.view(-1, tmp_concat_node_update.shape[2]))
        updated_node_embedding = F.relu(updated_node_embedding)
       
        aggregate_nodes, _ = torch_scatter.scatter_max(updated_node_embedding, batch_indices, dim=0)
       
        group_embedding, _ = self.to_global_nn(torch.cat([aggregate_nodes, global_attr], 2))
        group_embedding = F.relu(group_embedding)
        
        if self.second_layer:
            tmp_src_2 = torch.index_select(updated_node_embedding, 0, src_index) # for a in x_nodes ])
            
            tmp_target_2 = torch.index_select(updated_node_embedding, 0, target_index) #for a in x_nodes ])


            tmp_glob_2 = torch.repeat_interleave(group_embedding, num_edges, 0)


            tmp_concat_2 = torch.cat([tmp_src_2, message, tmp_target_2, tmp_glob_2], 2)
            

            message_tmp_2,_ = self.message_nn_2(tmp_concat_2)

            message_2 = F.relu(message_tmp_2) 
            
          
            
            target_index_expand_2 = target_index.expand((message_2.shape[1], target_index.size(0))).T
            
         
            output_tensor_aggr_tmp_2, _ =  torch_scatter.scatter_max(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            
            output_tensor_aggr_2 = output_tensor_aggr_tmp_2 
            
            tmp_glob_for_nodes_2 = torch.repeat_interleave(group_embedding, num_nodes, 0)
       
            tmp_concat_node_update_2 = torch.cat([updated_node_embedding, output_tensor_aggr_2, tmp_glob_for_nodes_2], 2)
          
            updated_node_embedding, _ = self.update_nn_2(tmp_concat_node_update_2)
            updated_node_embedding = F.relu(updated_node_embedding)

            aggregate_nodes,_ = torch_scatter.scatter_max(updated_node_embedding, batch_indices, dim=0)
            group_embedding, _ = self.to_global_nn_2(torch.cat([aggregate_nodes, group_embedding], 2))
            group_embedding = F.relu(group_embedding)

           
        results_per_node,_ = self.embedding_nn(updated_node_embedding)
       
        results_group, _ = self.group_adr_prediction(group_embedding)
        
        result_per_node = torch.squeeze(results_per_node[:,-1,:], dim=1)
        
        result_per_node = result_per_node.view(-1, num_nodes)
        
        results_group = results_group[:,-1,:]
        
        result = torch.cat([result_per_node, results_group], 1)

        return self.softmax(result) 
    

class SingleVectorNetwork(nn.Module):

    def __init__(self, n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes=4, lstm_second=False):
        super(SingleVectorNetwork, self).__init__()
        self.lstm_second = lstm_second
        self.first_layer = nn.LSTM(n_features_nodes*3+n_features_global,layer_1)
        if lstm_second:
            self.second_layer = nn.LSTM(layer_1, layer_2)

        self.third_layer = torch.nn.Sequential()
        layers = layer_linear.split('-')
        
        input_dim = layer_2 if lstm_second else layer_1
        for i in range(len(layers)):
            self.third_layer.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.third_layer.add_module("Relu"+str(i), nn.ReLU())
            
            input_dim = int(layers[i])
        self.third_layer.add_module("Lin"+str(len(layers)), nn.Linear(input_dim, int(num_classes)))
        self.softmax = nn.Softmax(dim=1)
        
        self.num_parameters = self.count_parameters(self)
        self.print_parameters()
    
    def print_parameters(self):
        print('First model:', self.count_parameters(self.first_layer ), self.first_layer)
        if self.lstm_second:
            print('Second model:', self.count_parameters(self.second_layer ), self.second_layer)
        print('Third model:', self.count_parameters(self.third_layer ), self.third_layer)
        print('Overall:', self.count_parameters(self ))

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 


    def forward(self, nodes, edge_indices, global_attr, num_nodes, num_edges, batch_indices, return_embeddings=False):
 
        global_attr = global_attr.view(-1, nodes.shape[1], global_attr.shape[1])
        nodes = nodes.view(-1, 3, nodes.shape[1], nodes.shape[2])
       
        nodes_concat = torch.cat([nodes[:,0,:,:],nodes[:,1,:,:], nodes[:,2,:,:]], 2)
        
        input_n = torch.cat([nodes_concat, global_attr], 2)
        
        output1, _ = self.first_layer(input_n)
        if self.lstm_second:
            output2, _ = self.second_layer(output1)
            output2 = output2[:,-1,:].view(output2.shape[0],-1)
        else:
            output2 = output1[:,-1,:].view(output1.shape[0],-1)
 
        result = self.third_layer(output2)
       
        return self.softmax(result) 

class SingleVectorFixedLengthNetwork(nn.Module):

    def __init__(self, n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes=2, lstm_second=False):
        super(SingleVectorFixedLengthNetwork, self).__init__()
        self.lstm_second = lstm_second
        self.first_layer = nn.LSTM(n_features_nodes*2+n_features_global,layer_1)
        if lstm_second:
            self.second_layer = nn.LSTM(layer_1, layer_2)

        self.third_layer = torch.nn.Sequential()
        layers = layer_linear.split('-')

        input_dim = layer_2 if lstm_second else layer_1
        for i in range(len(layers)):
            self.third_layer.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.third_layer.add_module("Relu"+str(i), nn.ReLU())
           
            input_dim = int(layers[i])

        self.third_layer.add_module("Lin"+str(len(layers)), nn.Linear(input_dim, int(num_classes)))
        self.softmax = nn.Softmax(dim=1)
       
        self.num_parameters = self.count_parameters(self)
        self.print_parameters()
    
    def print_parameters(self):
        print('First model:', self.count_parameters(self.first_layer ), self.first_layer)
        if self.lstm_second:
            print('Second model:', self.count_parameters(self.second_layer ), self.second_layer)
        print('Third model:', self.count_parameters(self.third_layer ), self.third_layer)
        print('Overall:', self.count_parameters(self ))

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 


    def forward(self, nodes, edge_indices, global_attr, num_nodes, num_edges, batch_indices, return_embeddings=False):
   
        global_attr = global_attr.view(-1, nodes.shape[1], global_attr.shape[1])

        tmp_glob = torch.repeat_interleave(global_attr, num_nodes, 0)
       
        input_n = torch.cat([nodes, tmp_glob], 2)

        output1, _ = self.first_layer(input_n)
        if self.lstm_second:
            output2, _ = self.second_layer(output1)
            output2 = output2[:,-1,:].view(output2.shape[0],-1)
        else:
            output2 = output1[:,-1,:].view(output1.shape[0],-1)
          
        result = self.third_layer(output2)
       
        return self.softmax(result) 

class SingleVectorFixedLengthNetworkType(nn.Module):

    def __init__(self, n_features_nodes, n_features_global, layer_1, layer_2, layer_linear, num_classes=10, lstm_second=False):
        super(SingleVectorFixedLengthNetworkType, self).__init__()
        self.lstm_second = lstm_second
        self.first_layer = nn.LSTM(n_features_nodes*2+n_features_global,layer_1)
        if lstm_second:
            self.second_layer = nn.LSTM(layer_1, layer_2)

        self.third_layer = torch.nn.Sequential()
        layers = layer_linear.split('-')

        input_dim = layer_2 if lstm_second else layer_1
        for i in range(len(layers)):
            self.third_layer.add_module("Lin"+str(i), nn.Linear(input_dim, int(layers[i])))
            self.third_layer.add_module("Relu"+str(i), nn.ReLU())
           
            input_dim = int(layers[i])
       
        self.third_layer.add_module("Lin"+str(len(layers)), nn.Linear(input_dim, int(num_classes)))
        self.softmax = nn.Softmax(dim=1)
        
        self.num_parameters = self.count_parameters(self)
        self.print_parameters()
    
    def print_parameters(self):
        print('First model:', self.count_parameters(self.first_layer ), self.first_layer)
        if self.lstm_second:
            print('Second model:', self.count_parameters(self.second_layer ), self.second_layer)
        print('Third model:', self.count_parameters(self.third_layer ), self.third_layer)
        print('Overall:', self.count_parameters(self ))

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    
    def forward(self, nodes, edge_indices, global_attr, num_nodes, num_edges, batch_indices, return_embeddings=False):
  
        global_attr = global_attr.view(-1, nodes.shape[1], global_attr.shape[1])
    
        input_n = torch.cat([nodes, global_attr], 2)
       
        output1, _ = self.first_layer(input_n)
        if self.lstm_second:
            output2, _ = self.second_layer(output1)
            output2 = output2[:,-1,:].view(output2.shape[0],-1)
        else:
            output2 = output1[:,-1,:].view(output1.shape[0],-1)
           
        result = self.third_layer(output2)
       
        return self.softmax(result) 
 


class MMPNType(nn.Module):

    def __init__(self, n_features_nodes, n_features_global, message_arch, node_arch, n_embedding_group, n_output_dim_node, n_output_dim_action, 
                    second_layer=False, second_message_arch='', second_node_update_arch='', 
                    third_layer=False, third_message_arch='', third_node_update_arch='', memory_network_block=nn.LSTM):
        super(MMPNType, self).__init__()
  
        n_features_edges = 0
        self.second_layer = second_layer
        self.third_layer = third_layer

        self.message_nn = torch.nn.Sequential()
        layers = message_arch.split('-')
        n_message_dim = int(layers[-1])
        input_dim = n_features_edges+n_features_nodes+n_features_nodes+n_features_global
        for i in range(len(layers)):
            self.message_nn.add_module("LSTM"+str(i), memory_network_block(input_dim, int(layers[i])))
            input_dim = int(layers[i])

        print('Input dim message', n_features_edges+n_features_nodes+n_features_nodes+n_features_global)
        
        self.update_nn = torch.nn.Sequential()
        layers = node_arch.split('-')
        input_dim = n_message_dim + n_features_nodes + n_features_global
        n_embedding_dim = int(layers[-1])
        print(layers, n_embedding_dim)
        for i in range(len(layers)):
            self.update_nn.add_module("LSTM"+str(i), memory_network_block(input_dim, int(layers[i])))
            input_dim = int(layers[i])


        # update global
        self.to_global_nn = memory_network_block(n_embedding_dim+n_features_global, n_embedding_group)
        
        if second_layer:
            # second round of GNN
           
            self.message_nn_2 = torch.nn.Sequential()
            layers = second_message_arch.split('-')
            n_message_dim_2 = int(layers[-1])
            print(layers, n_message_dim_2)
            print("Second message dim", n_message_dim_2)
            input_dim = n_message_dim+n_embedding_dim+n_embedding_dim+n_embedding_group
            for i in range(len(layers)):
                self.message_nn_2.add_module("LSTM"+str(i), memory_network_block(input_dim, int(layers[i])))
                input_dim = int(layers[i])


            self.update_nn_2 = torch.nn.Sequential()
            layers = second_node_update_arch.split('-')
            
            print('Input to second update', n_message_dim_2, n_embedding_dim, n_embedding_group)
            input_dim = n_message_dim_2+n_embedding_dim + n_embedding_group
            n_embedding_dim = int(layers[-1])
            for i in range(len(layers)):
                self.update_nn_2.add_module("LSTM"+str(i), memory_network_block(input_dim, int(layers[i])))
                input_dim = int(layers[i])
            
            self.to_global_nn_2 = memory_network_block(n_embedding_dim+n_embedding_group, n_embedding_group)

        


        self.action_prediction = memory_network_block(n_embedding_dim+n_embedding_group, n_output_dim_action)
        print('Size input = ', n_embedding_dim, '+', n_embedding_group)
        self.softmax = nn.Softmax(dim=1)
        
        
        self.print_parameters()
    
    def print_parameters(self):
        print('Message model:', self.count_parameters(self.message_nn ), self.message_nn)
        print('Update model:', self.count_parameters(self.update_nn ), self.update_nn)
        print("Global model", self.count_parameters(self.to_global_nn ), self.to_global_nn)
        if self.second_layer:
            print('Message model 2:', self.count_parameters(self.message_nn_2 ), self.message_nn_2)
            print('Update model 2:', self.count_parameters(self.update_nn_2 ), self.update_nn_2)
            print("Global model 2", self.count_parameters(self.to_global_nn_2 ), self.to_global_nn_2)
        if self.third_layer:
            print('Message model 3:', self.count_parameters(self.message_nn_3 ), self.message_nn_3)
            print('Update model 3:', self.count_parameters(self.update_nn_3 ), self.update_nn_3)
            print("Global model 3", self.count_parameters(self.to_global_nn_3 ), self.to_global_nn_3)
        print('Last layer', self.count_parameters(self.action_prediction ), self.action_prediction)
        print('Overall:', self.count_parameters(self ))

    def count_parameters(self, model): 
        return sum(p.numel() for p in model.parameters() if p.requires_grad) 

    
    def forward(self, nodes, edge_indices, global_attr, num_nodes, num_edges, batch_indices, chosen_who):
        

        actions_target_batch_wide = torch.clone(chosen_who).cpu()
        elements_unique, counts = np.unique(batch_indices, return_counts=True)
        for i in range(1, chosen_who.shape[0]):
            
            if actions_target_batch_wide[i] == 3:
                actions_target_batch_wide[i] -= 1
            actions_target_batch_wide[i] = actions_target_batch_wide[i] + np.sum(counts[:i])

        edges_list_index_to_list = []
        overall_count = 0
       
        src_index = edge_indices[0] 
        target_index = edge_indices[0]
        
        x_nodes = nodes
        
        tmp_src = torch.index_select(x_nodes, 0, src_index) 
        
        tmp_target = torch.index_select(x_nodes, 0, target_index) 
        
   
        
        global_attr = global_attr.view(-1, x_nodes.shape[1], global_attr.shape[1])
        
        tmp_glob = torch.repeat_interleave(global_attr, num_edges, 0)
     
        tmp_concat = torch.cat([tmp_src, tmp_target, tmp_glob], 2)

        message_result, _ = self.message_nn(tmp_concat)
        

        message = F.relu(message_result)
        
        # resort to edges to batches
        
        
        target_index_expand = target_index.expand((message.shape[1], target_index.size(0))).T
       
        output_tensor_aggr_tmp, _ =  torch_scatter.scatter_min(message, target_index_expand, dim=0) #for a in torch.Tensor(message) ])
        
        output_tensor_aggr = output_tensor_aggr_tmp 
        
        tmp_glob_for_nodes = torch.repeat_interleave(global_attr, num_nodes, 0)
        
        tmp_concat_node_update = torch.cat([x_nodes, output_tensor_aggr, tmp_glob_for_nodes], 2)
        
        updated_node_embedding, _ = self.update_nn(tmp_concat_node_update) #.view(-1, tmp_concat_node_update.shape[2]))
        
        updated_node_embedding = F.relu(updated_node_embedding)
        
        aggregate_nodes,_ = torch_scatter.scatter_min(updated_node_embedding, batch_indices, dim=0)
        
        group_embedding, _ = self.to_global_nn(torch.cat([aggregate_nodes, global_attr], 2))
        group_embedding = F.relu(group_embedding)
        
        if self.second_layer:
            tmp_src_2 = torch.index_select(updated_node_embedding, 0, src_index) # for a in x_nodes ])
            
            tmp_target_2 = torch.index_select(updated_node_embedding, 0, target_index) #for a in x_nodes ])


            tmp_glob_2 = torch.repeat_interleave(group_embedding, num_edges, 0)

            
            tmp_concat_2 = torch.cat([tmp_src_2, message, tmp_target_2, tmp_glob_2], 2)
            

            
            message_tmp_2, _ = self.message_nn_2(tmp_concat_2)
            
            
            message_2 = F.relu(message_tmp_2) 
            

            
            target_index_expand_2 = target_index.expand((message_2.shape[1], target_index.size(0))).T
            
           
            output_tensor_aggr_tmp_2, _ =  torch_scatter.scatter_min(message_2, target_index_expand_2, dim=0) #for a in torch.Tensor(message) ])
            
            output_tensor_aggr_2 = output_tensor_aggr_tmp_2 
           
            tmp_glob_for_nodes_2 = torch.repeat_interleave(group_embedding, num_nodes, 0)
           

        
            tmp_concat_node_update_2 = torch.cat([updated_node_embedding, output_tensor_aggr_2, tmp_glob_for_nodes_2], 2)
        
            updated_node_embedding, _ = self.update_nn_2(tmp_concat_node_update_2)
            updated_node_embedding = F.relu(updated_node_embedding)

            aggregate_nodes = torch_scatter.scatter_mean(updated_node_embedding, batch_indices, dim=0)
           
            group_embedding, _ = self.to_global_nn_2(torch.cat([aggregate_nodes, group_embedding], 2))
            group_embedding = F.relu(group_embedding)


        aggregate_nodes_for_all,_ = torch_scatter.scatter_min(updated_node_embedding, batch_indices, dim=0)
        
        chosen_nodes = torch.index_select(updated_node_embedding, 0, actions_target_batch_wide)
        
        condition = (chosen_who == 3)
        


        chosen_nodes[condition,:,:] = aggregate_nodes_for_all[condition,:,:]
        tmp_concat_action_pred = torch.cat([chosen_nodes, group_embedding], 2)

        

        results_actions, _ = self.action_prediction(tmp_concat_action_pred)
        results_actions = results_actions
        
        results_actions = results_actions[:,-1,:]
        
        return self.softmax(results_actions)
