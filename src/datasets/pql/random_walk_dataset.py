from ..random_walk_dataset import RandomWalkDataset
import pandas as pd
class RandomWalkPQLDataset(RandomWalkDataset):
    """Gets the Knowledge Graph as an input and should create Random Walks. For each node sample up to 20 random walks of length 3, do this 5 times with different seeds.
    #Hold out the random walks which are triples in the validation and test set.

    Returns: Random Walk dataset should output an incomplete sequence like "e1 ; r1 ; r2 ; ... ; rn-1;" and a complete sequence "e1 ; r1 ; e2 ; ... ; rn-1 ; en"
    """
           
    def _get_walks(self, dataset):
        """
        Extract walks from the dataset's evidences.
        
        Parameters:
        - dataset (DataFrame): DataFrame containing a column 'evidences' where each entry is a list of evidence tuples.

        Returns:
        - List of walks as tuples.
        """
        walks = set() 
        for entry in dataset['evidences']:
            tuple_evidence = tuple(entry)
            walks.add(tuple_evidence)
        return list(walks)
    