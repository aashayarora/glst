from argparse import ArgumentParser
import json

import torch
torch.manual_seed(42)
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

from model import GravNetGNN
from dataset import GraphDataset
from train import train, validation
from helpers import plot_loss

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

from fastgraphcompute.object_condensation import ObjectCondensation

from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--config', type=str, default='config.json', help='Path to the config file')
    argparser.add_argument('--debug', action='store_true', help='debug mode')
    argparser.add_argument('--validation_only', action='store_true', help='Run validation only, skip training')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    input_path = config.get('input_path', '../data/')
    output_path = config.get('output_path', './')
    regex = config.get('regex', '*.pt')
    
    epochs = config.get('epochs', 50)
    validation_step = config.get('validation_step', 5)
    save_step = config.get('save_step', 100)
    
    batch_size = config.get('batch_size', 16)
    
    train_size = config.get('training_split', 0.8)
    learning_rate = config.get('learning_rate', 0.001)

    subset = config.get('subset', None)
    if args.debug:
        subset = 10

    transform = None

    dataset = GraphDataset(input_path=input_path, regex=regex, subset=subset, transform=transform)

    train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, random_state=42)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=8, drop_last=True, num_workers=4, pin_memory=True)    

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_node_features = dataset[0].num_node_features

    model = GravNetGNN(
        in_dim=num_node_features,
        k=12
    )
    model.to(device)

    if not args.validation_only:
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = ObjectCondensation(q_min=0.5, s_B=1)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

        tot_losses = {}
        for epoch in tqdm(range(1, epochs+1)):
            model_out, losses = train(model, device, optimizer, criterion, scheduler, train_loader)
            for key, value in losses.items():
                if key not in tot_losses:
                    tot_losses[key] = []
                tot_losses[key].append(value)
            tqdm.write(f"Epoch {epoch}/{epochs} - Losses: {losses}")

            if epoch > 0 and epoch % save_step == 0:
                torch.save(model.state_dict(), f"models/model_{epoch}.pt")
                tqdm.write(f"Model saved at epoch {epoch}")
                plot_loss(tot_losses)

    else:
        model.load_state_dict(torch.load("models/model_200.pt", weights_only=False))
        model.eval()
        
    eps_range = np.arange(0.01, 0.05, 0.01)
    perfect_rates, lhc_rates, dm_rates = [], [], []
    
    for eps in tqdm(eps_range, desc="Testing epsilon values"):
        perfect_batch, lhc_batch, dm_batch = [], [], []
        
        for data in test_loader:
            data = data.to(device)
            perf, lhc, dm = validation(data, model, device, eps=eps)
            perfect_batch.append(perf)
            lhc_batch.append(lhc)
            dm_batch.append(dm)

        print(perfect_batch)
        
        perfect_rates.append(np.mean(perfect_batch))
        lhc_rates.append(np.mean(lhc_batch))
        dm_rates.append(np.mean(dm_batch))
    
    fig, ax = plt.subplots()
    ax.plot(eps_range, perfect_rates, 'o-', label='Perfect')
    ax.plot(eps_range, lhc_rates, 's-', label='LHC')
    ax.plot(eps_range, dm_rates, '^-', label='DM')
    
    ax.set_xlabel('Epsilon (Clustering Distance Threshold)')
    ax.set_ylabel('Rate')
    ax.legend()
    
    plt.savefig("plots/eps_test.png")
    plt.close()