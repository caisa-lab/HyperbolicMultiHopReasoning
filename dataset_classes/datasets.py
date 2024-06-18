from torch.utils.data import Dataset
import random
import networkx as nx 
import time

class RandomWalkDataset(Dataset):
    """Gets the Knowledge Graph as an input and should create Random Walks.

    Returns: Random Walk dataset should output an incomplete sequence like "e1 ; r1 ; r2 ; ... ; rn-1;" and a complete sequence "e1 ; r1 ; e2 ; ... ; rn-1 ; en"
    """
    def __init__(self, kg, steps):
        self.kg = kg
        self.steps = steps
        
        random_paths = []
        index = 1
        for _ in range(5):
            random.seed(index)
            for start_node in list(self.kg.nodes()):
                random_paths += self._create_random_walks(self.kg, start_node, self.steps, tries=20) 
            index += 1
            
        self.data = list(set(random_paths))
        
    
    def _create_random_walks(self, graph, start_node, number_of_steps, tries = 30):
        random_paths = []
        for _ in range(tries):
            current_node = start_node
            walk = [current_node]
            for _ in range(number_of_steps):
                neighbors = list(graph.successors(current_node))
                if not neighbors:
                    break  # Restart the walk if dead-end is reached
                next_node = random.choice(neighbors)
                edge = graph.get_edge_data(current_node, next_node)['relation']
                walk.append(edge)
                walk.append(next_node)
                current_node = next_node
            if len(walk) == 2 * self.steps - 1:
                random_paths.append(tuple(walk))
        return random_paths
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        random_walk = self.data[idx]
        
        
        complete_path = f"{random_walk[0]}"
        incomplete_path = f"{random_walk[0]}"
        for i in range(1, len(random_walk)):
            complete_path += f" ; {random_walk[i]}"
            if i % 2 != 0:
                incomplete_path += f" ; {random_walk[i]}"

        
        return complete_path, incomplete_path


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
        
