from torch.utils.data import Dataset

class KnowledgeIntegrationMusiqueDataset(Dataset):
    """
    Needs a dataset that has all entries so train + dev + test. 
    """
    def __init__(self, dataset_with_all_entries):
        self.dataset = dataset_with_all_entries
        unique_subquestions_answer_pairs = set()
        for decomposition in dataset_with_all_entries['question_decomposition']:
            for item in decomposition:
                unique_subquestions_answer_pairs.add((item['question'], item['answer']))
        
        self.data = list(unique_subquestions_answer_pairs)
    def __len__(self):
        return len(self.data)
    
    def __getitem__ (self, idx):
        subquestion, answer = self.data[idx]
        input_str = f"{subquestion}"
        label = f"{answer}"
        
        return input_str, label