from torch.utils.data import Dataset
import random
import networkx as nx 
import time
from tqdm import tqdm
import pandas as pd

class RandomWalkDataset(Dataset):
    """Gets the Knowledge Graph as an input and should create Random Walks. For each node sample up to 20 random walks of length 3, do this 5 times with different seeds.
    Hold out the random walks which are triples in the validation and test set.

    Returns: Random Walk dataset should output an incomplete sequence like "e1 ; r1 ; r2 ; ... ; rn-1;" and a complete sequence "e1 ; r1 ; e2 ; ... ; rn-1 ; en"
    """
    def __init__(self, kg, steps):
        self.kg = kg
        self.steps = steps

        self.data = list(self._generate_random_walks())
        
    def _random_walk(self, start_node, hops=2):
        path = [start_node]
        current_node = start_node
        for _ in range(hops):
            neighbors = list(self.kg.successors(current_node))
            if not neighbors:
                break  # Restart the walk if dead-end is reached
            next_node = random.choice(neighbors)
            edge = self.kg.get_edge_data(current_node, next_node)['relation']
            path.append(edge)
            path.append(next_node)
            current_node = next_node
        return path
    
    def _generate_random_walks(self, num_walks_per_node=30, walk_length=3, num_iterations=10):
        all_paths = set()
        nodes = list(self.kg.nodes())
        for _ in tqdm(range(num_iterations)):
            random.seed()  # Change seed for each iteration
            for idx, node in enumerate(nodes):
                walks = set()
                for _ in range(num_walks_per_node):
                    path = self._random_walk(node, walk_length-1)
                    if path and len(path) == 2*walk_length - 1:  # Ensure path is exactly of the required length
                        walks.add(tuple(path))
                all_paths.update(walks)
            #print(len(list(all_paths)))
        return all_paths
    
    
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
    """Single Hop Dataset gets a dataset

    Returns:
        Returns a random single hop containing input_str = "e1 ; relation" and label "e2"
    """
    def __init__(self, dataset, triples_not_to_contain = set()):
        self.dataset = dataset
        self.triples_not_to_contain = triples_not_to_contain
        self.relation_question_mapping = {
            'director': ['Who is the director of X?', 'Who directed the film X?'],
            'date of birth': ['What is the date of birth of X?', 'When is X’s birthday?', 'When was X born?'],
            'date of death': ['When did X die?', 'What is the date of death of X?'],
            'country': ['What country is X from?', 'What is thenationality of X?'],
            'country of citizenship': ['What country is X from?', 'What is the nationality of X?'],
            'award received': ['What is the award that X received?', 'Which award did X receive?'],
            'cause of death': ['Why did X die?', 'What was the causeof X’s death?'],
            'composer': ['Who is the composer of X?', 'Who composed X?'],
            'creator': ['Who is the creator of X?', 'Who created X?'],
            'child': ['Who is the child of X?'],
            'doctoral advisor': ['Who is the doctoral advisor of X?'],
            'editor': ['Who is the editor of X?', 'Who edited X?'],
            'educated at': ['Where did X graduate from?', 'What is the alma mater of X?', 'Where did X study?'],
            'employer': ['Who is the employer of X?', 'Where does X work?'],
            'father': ['Who is the father of X?', 'Who is X’s father?'],
            'mother': ['Who is the mother of X?', 'Who is X’s mother?'],
            'founded by': ['Who is the founder of X?', 'Who founded X?'],
            'inception': ['When was X founded?'],
            'manufacturer': ['Who manufactures X?'],
            'performer': ['Who is the performer of the song X?', 'Who performed the song X?'],
            'place of birth': ['Where was X born?', 'What is the place of birth of X?'],
            'place of burial': ['Where was X buried?', 'Where is the place of burial of X?'],
            'place of death': ['Where did X die?', 'Where is the place of death of X?'],
            'place of detention': ['Where did X go to prison?', 'Where was X detained?'],
            'presenter': ['Who is the presenter of X?', 'Who presented X?'],
            'publisher': ['Who published X?', 'What company published X?'],
            'sibling': ['Who is the sibling of X?', 'Who is X’s sibling?'],
            'spouse': ['Who is the spouse of X?', 'Who is X’s spouse?'],
            'student of': ['Who was the teacher of X?', 'Who was X’s teacher?'],
        }
        
        
        
        self.data = self._create_single_hop_dataset(dataset)
    
    def _create_single_hop_dataset(self, dataset):
        unique_triples = set()
        for entry in dataset['evidences']:
            for triples in entry:
                unique_triples.add(tuple(triples))
        
        
        rows = []
        unique_id = 0 
        for triple in unique_triples:
            if triple in self.triples_not_to_contain:
                continue
            e1, relation, e2 = triple[0], triple[1], triple[2]
            possible_questions = self.relation_question_mapping[relation]
            random_question = random.choice(possible_questions)
            random_question = random_question.replace('X', e1) # Replace the X with the entity

            rows.append({
                '_id': unique_id, #Just a Counter to give unique ids
                'type': 'single_hop_question',
                'question': random_question, # Random Question from the Dictionary
                'answer': e2, # Label
                'context': 0, # TODO Need to add the context. Take the relevant context parts of the Multi Hop Question.
                'evidences': triple # Single Hop has only one evidence
            })
            
            unique_id += 1
        single_hop_dataset = pd.DataFrame(rows,columns=['_id', 'type', 'question', 'context', 'evidences', 'answer'])
        return single_hop_dataset
                
                   
                
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx]

class OneWikiHopDataset(Dataset):
    """
    Takes all Datasets and creates the OneWikiHopDataset for the type. Returns 
    {
            'question': question,
            'context': context,
            'evidences': evidences,
            'answer': answer
        }
    
    """
    def __init__(self, train_dataset, dev_dataset, test_dataset, type='train'):
        if type not in ['train', 'test', 'dev']:
            raise ValueError(f"Unknown type: {type}. Supported Types are ['train', 'test', 'dev']")
        self.type = type
        triples_not_to_contain = set()
        if type == 'train':
            self.dataset = SingleHopDataset(train_dataset, triples_not_to_contain)
        elif type == 'dev':
            single_hop_dataset_train = SingleHopDataset(train_dataset, triples_not_to_contain)
            for triple in single_hop_dataset_train.data['evidences']:
                triples_not_to_contain.add(triple)
            self.dataset = SingleHopDataset(dev_dataset, triples_not_to_contain)
        elif type == 'test':
            single_hop_dataset_train = SingleHopDataset(train_dataset, triples_not_to_contain)
            for triple in single_hop_dataset_train.data['evidences']:
                triples_not_to_contain.add(triple)
            single_hop_dataset_dev = SingleHopDataset(dev_dataset, triples_not_to_contain)
            for triple in single_hop_dataset_dev.data['evidences']:
                triples_not_to_contain.add(triple)
            self.dataset = SingleHopDataset(test_dataset, triples_not_to_contain)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        question = self.dataset['question'][idx]
        context = self.dataset['context'][idx]
        evidences = self.dataset['evidences'][idx]
        answer = self.dataset['answer'][idx]
        
        return {
            'question': question,
            'context': context,
            'evidences': evidences,
            'answer': answer
        }
           
class KnowledgeIntegrationDataset(Dataset):
    """
    Needs a dataset that has all entries so train + dev + test
    """
    def __init__(self, dataset_with_all_entries):
        self.dataset = dataset_with_all_entries
        unique_triples = set()
        for entry in dataset_with_all_entries['evidences']:
            for triple in entry:
                unique_triples.add(tuple(triple))
        
        self.data = list(unique_triples)
    def __len__(self):
        return len(self.data)
    def __getitem__ (self, idx):
        e1, relation, e2 = self.data[idx]
        input_str = f"{e1} ; {relation}"
        label = f"{e2}"
        
        return input_str, label
