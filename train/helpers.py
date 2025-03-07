import numpy as np
import torch
from torch_geometric.transforms import BaseTransform

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

import mplhep as hep
hep.style.use(hep.style.CMS)
import matplotlib.pyplot as plt

class PerformanceEvaluator:
    def __init__(self, model, device, train_loader, test_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        train_predictions = []
        train_labels = []
        test_predictions = []
        test_labels = []
        self.model.eval()
        with torch.no_grad():
            for data in self.train_loader:
                x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
                x, edge_index, edge_attr, y = [i.to(self.device) for i in (x, edge_index, edge_attr, y)]
                pred = self.model(x, edge_index, edge_attr).squeeze()
                train_predictions.append(pred)
                train_labels.append(y)

            for data in self.test_loader:
                x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
                x, edge_index, edge_attr, y = [i.to(self.device) for i in (x, edge_index, edge_attr, y)]
                pred = self.model(x, edge_index, edge_attr).squeeze()
                test_predictions.append(pred)
                test_labels.append(y)

        self.train_predictions = torch.cat(train_predictions).cpu().numpy()
        self.train_labels = torch.cat(train_labels).cpu().numpy()
        self.test_predictions = torch.cat(test_predictions).cpu().numpy()
        self.test_labels = torch.cat(test_labels).cpu().numpy()

    def plot_precision_recall_curve(self, save_path='plots/precision_recall_curve.png'):
        self.train_precision, self.train_recall, _ = precision_recall_curve(self.train_labels, self.train_predictions)
        self.test_precision, self.test_recall, _ = precision_recall_curve(self.test_labels, self.test_predictions)

        fig, ax = plt.subplots()
        ax.plot(self.train_recall[5:], self.train_precision[5:])
        ax.plot(self.test_recall[5:], self.test_precision[5:])
        ax.set_xlabel('Recall (Efficiency)')
        ax.set_ylabel('Precision (1 - Fake Rate)')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='upper right')

        plt.savefig(save_path)

    def plot_roc_curve(self, save_path='plots/roc_curve.png'):
        self.train_fpr, self.train_tpr, _ = roc_curve(self.train_labels, self.train_predictions)
        self.train_auc = roc_auc_score(self.train_labels, self.train_predictions)
        self.test_fpr, self.test_tpr, _ = roc_curve(self.test_labels, self.test_predictions)
        self.test_auc = roc_auc_score(self.test_labels, self.test_predictions)

        fig, ax = plt.subplots()
        ax.plot(self.train_fpr, self.train_tpr, label=f'Train AUC = {self.train_auc:.2f}')
        ax.plot(self.test_fpr, self.test_tpr, label=f'Test AUC = {self.test_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='upper right')

        plt.savefig(save_path)


def plot_loss(train_loss, test_loss, test_step, save_path='plots/loss_plot.png'):
    fig, ax = plt.subplots()
    train_x = np.arange(len(train_loss))
    test_x = np.arange(0, len(test_loss) * test_step, test_step)
    ax.plot(train_x, train_loss, label='Train Loss')
    ax.plot(test_x, test_loss, label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    plt.savefig(save_path)

class ColumnWiseNormalizeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'x') and data.x is not None:
            mean = data.x.mean(dim=0, keepdim=True) 
            std = data.x.std(dim=0, keepdim=True) 
            
            std[std == 0] = 1  
            
            data.x = (data.x - mean) / std 

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            mean = data.edge_attr.mean(dim=0, keepdim=True)
            std = data.edge_attr.std(dim=0, keepdim=True)

            std[std == 0] = 1
            
            data.edge_attr = (data.edge_attr - mean) / std
            
        return data
