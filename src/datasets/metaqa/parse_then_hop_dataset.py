from torch.utils.data import Dataset
import pandas as pd
import pandas as pd
from torch.utils.data import Dataset

import pandas as pd
from torch.utils.data import Dataset

class ParseMetaQADataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_answers : int = 3):
        """
        Initializes the dataset with the provided DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing 'question' and 'evidences' columns.
        """
        # Ensure the required columns exist
        required_columns = {'question', 'evidences'}
        if not required_columns.issubset(dataframe.columns):
            missing = required_columns - set(dataframe.columns)
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        
        # Select only the required columns and reset index
        self.dataset = dataframe[['question', 'evidences']].reset_index(drop=True)
        self.dataset = self.dataset[self.dataset['evidences'].apply(lambda x : len(x) <= max_answers)]

        for evidence_list in self.dataset['evidences']:
            if len(evidence_list) > max_answers:
                print(f"Found List with more than {max_answers} evidences: {evidence_list}")
        print(f"No Evidence List found with more than {max_answers}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Retrieves the question and its corresponding processed incomplete evidences.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the question (str) and the processed incomplete evidences (str).
        """
        if not 0 <= idx < len(self.dataset):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.dataset)}.")
        
        # Retrieve the row at the specified index
        row = self.dataset.iloc[idx]
        question = row['question']
        evidences = row['evidences']
        
        # Process the evidences to retain only e1 and all relations
        incomplete_evidences = self.construct_incomplete_paths(evidences)
        
        return question, incomplete_evidences

    @staticmethod
    def construct_incomplete_paths(evidences):   
        evidence_list = evidences[0] 
        # Construct incomplete path for the current evidence list
        incomplete_path = f"{evidence_list[0]}"  # Start with the first entity
        for i in range(1, len(evidence_list)):
            if i % 2 != 0:  # Add only relations (odd indices)
                incomplete_path += f" ; {evidence_list[i]}"
            

        
        return incomplete_path




    
class ParseThenHopMusiqueDataset(Dataset):
    def __init__(self, dataframe : pd.DataFrame, max_answers : int = 3):
        self.dataset = dataframe[['question', 'evidences']]
        self.dataset = self.dataset[self.dataset['evidences'].apply(lambda x : len(x) <= max_answers)]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__ (self, idx):
        question = self.dataset.iloc[idx]['question']
        evidences = self.dataset.iloc[idx]['evidences']
        
        complete_path = self._process_evidences(evidences)

            
        return question, complete_path
    
    @staticmethod
    def _process_evidences(evidences):
        """
        Processes the 'evidences' to create incomplete sequences.

        Args:
            evidences (list): A list of complete sequences, where each complete sequence is a list in the format 
                              [e1, r1, e2, r2, ..., rn-1, en].

        Returns:
            str: A string representing the incomplete sequences, formatted as 
                 "e1 ; r1 ; e2 ; r2 ... ; rn-1 | e1 ; r1 ; ... | ...".
        """
        if not isinstance(evidences, list):
            raise TypeError("Each 'evidences' entry must be a list of sequences.")
        
        complete_sequences = []
        
        for seq_idx, sequence in enumerate(evidences):
            if not isinstance(sequence, list):
                raise TypeError(f"Sequence at index {seq_idx} is not a list.")
            
            if len(sequence) < 2:
                # A valid complete sequence should have at least one entity and one relation
                raise ValueError(f"Sequence at index {seq_idx} is too short to process.")
            
            # Remove the last entity to create an incomplete sequence
            complete = sequence
            
            # Join the elements with ' ; '
            complete_str = ' ; '.join(map(str, complete))
            
            complete_sequences.append(complete_str)
        
        # Join all incomplete sequences with ' | '
        final_incomplete_sequence = ' | '.join(complete_sequences)
        
        return final_incomplete_sequence