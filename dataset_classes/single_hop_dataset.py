from torch.utils.data import Dataset
import random

class SingleHopDataset(Dataset):
    """Single Hop Dataset gets a knowledge graph

    Returns:
        Returns a random single hop containing input_str = "e1 ; relation" and label "e2"
    """
    def __init__(self, kg):
        self.kg = kg
        self.data = self._create_single_hop_dataset(self.kg)
        
    def _create_single_hop_dataset(self, graph):
        data = []
        for start_node in list(graph.nodes()):
            neighbors = list(self.kg.successors(start_node))
            if len(neighbors) == 0:
                continue
            for _ in range(20):
                next_node = random.choice(neighbors)
                relation = self.kg.get_edge_data(start_node, next_node)['relation']
                
                data.append((start_node, relation, next_node))
                if len(neighbors) == 1:
                    break
                
        return list(set(data))
                
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        (e1, r, e2) = self.data[idx]
            
        input_str = f"{e1} ; {r}"
        label = f"{e2}"
        
        return input_str, label
        
