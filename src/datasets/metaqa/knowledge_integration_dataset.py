from torch.utils.data import Dataset
import networkx as nx

class KnowledgeIntegrationMetaQADataset(Dataset):
    """
    Needs a dataset that has all entries so train + dev + test. 
    """
    def __init__(self, kg : nx.MultiGraph, undirected = True):
        self.data = []

        for u, v, data in kg.edges(data = True):
            rel = data.get('relation', 'unknown_relation')
            self.data.append((f"{u} ; {rel}", f"{v}"))
            if undirected:
                self.data.append((f"{v} ; {rel}", f"{u}"))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__ (self, idx):
        input_str, label = self.data[idx]
        return input_str, label