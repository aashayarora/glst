from torch_geometric.data import Dataset
import torch
from glob import glob

class GraphDataset(Dataset):
    def __init__(self, input_path, regex, subset=None, transform=None):
        super().__init__()
        self.subset = subset
        self.files = glob(input_path + regex)
        self.transform = transform

    def __len__(self):
        if self.subset is not None:
            return min(self.subset, len(self.files))
        return len(self.files)

    def __getitem__(self, idx):
        graph = torch.load(self.files[idx], weights_only=False)
        if self.transform:
            return self.transform(graph)
        return graph