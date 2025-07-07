import torch
import numpy as np

from sklearn.cluster import DBSCAN

from fastgraphcompute.torch_geometric_interface import row_splits_from_strict_batch as batch_to_rowsplits

def train(model, device, optimizer, criterion, scheduler, train_loader):
    model.train()
    tot_attractive_loss = 0
    tot_repulsive_loss = 0
    tot_loss = 0
    
    for i, data in enumerate(train_loader):
        data = data.to(device)
        row_splits = batch_to_rowsplits(data.batch)

        optimizer.zero_grad()
        model_out = model(data, row_splits)

        L_att, L_rep, L_beta, _, _ = criterion(
            beta = model_out["B"],
            coords = model_out["H"],
            asso_idx = data.y.to(torch.int32),
            row_splits = row_splits
        )
    
        tot_loss_batch = L_att + L_rep + L_beta

        tot_loss_batch.backward()
        optimizer.step()

        tot_attractive_loss += L_att.item()
        tot_repulsive_loss += L_rep.item()
        tot_noise_loss = L_beta.item()
        tot_loss += tot_loss_batch.item()

    losses = {
        "attractive": tot_attractive_loss / len(train_loader),
        "repulsive": tot_repulsive_loss / len(train_loader),
        "noise": tot_noise_loss / len(train_loader),
        "loss": tot_loss / len(train_loader),
    }
    
    if scheduler is not None:
        scheduler.step()

    return model_out, losses

@torch.no_grad()
def validation(data, model, device, eps=0.2):
    data = data.to(device)
    row_splits = batch_to_rowsplits(data.batch)

    model.eval()
    out = model(data, row_splits)
    
    X = out["H"].cpu().detach().numpy()
    cluster = DBSCAN(eps=eps, min_samples=2).fit(X)

    data_labels = data.y.cpu().detach().numpy().flatten()
    uniq_data_labels = sorted(set(data_labels))

    perfect = []
    lhc = []
    dm = []
    perf_truth = []
    perf_fakes = []

    for uniq_cl in uniq_data_labels:
        true_cluster_indices = np.where(data_labels == uniq_cl)[0]

        cluster_dbscan_labels = cluster.labels_[true_cluster_indices]
        
        if np.all(cluster_dbscan_labels == -1):
            continue

        non_noise_labels = cluster_dbscan_labels[cluster_dbscan_labels != -1]
        if len(non_noise_labels) == 0:
            continue  # Skip if all points are noise
        
        unique_labels, counts = np.unique(non_noise_labels, return_counts=True)
        dbscan_label = unique_labels[np.argmax(counts)]

        num_elements_true = len(true_cluster_indices)
        num_elements_pred = np.sum(cluster.labels_ == dbscan_label)
        num_elements_correct = np.sum(data_labels[cluster.labels_ == dbscan_label] == uniq_cl)
        num_elements_fake = num_elements_pred - num_elements_correct
        
        if num_elements_true > 0 and num_elements_pred > 0:
            perfect.append(num_elements_correct == num_elements_true)
            lhc.append(1 if num_elements_correct / num_elements_true > 0.75 else 0)
            dm.append(1 if (num_elements_correct / num_elements_true >= 0.5 and 
                           num_elements_fake / num_elements_pred < 0.5) else 0)
            
            perf_truth.append(num_elements_correct / num_elements_true)
            perf_fakes.append(num_elements_fake / num_elements_pred)
            
            print(f"Cluster {uniq_cl}: {num_elements_true} true elements, {num_elements_pred} identified elements, {num_elements_correct} correct, {num_elements_fake} fake")
    
    print(f"Number of true clusters: {len(uniq_data_labels)}")
    print(f"Number of predicted clusters: {len(set(cluster.labels_)) - 1}")
    print(f"Number of noisy segments: {np.sum(cluster.labels_ == -1)}")

    return np.mean(perfect), np.mean(lhc), np.mean(dm)
