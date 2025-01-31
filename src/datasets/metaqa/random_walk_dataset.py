from collections import defaultdict
from ..random_walk_dataset import RandomWalkDataset
class RandomWalkMetaQADataset(RandomWalkDataset):
    """
    Generates random walks from a Knowledge Graph, organizing them into incomplete paths with multiple completions.

    Each sample consists of:
    - Incomplete Path: "entity1 ; relation1 ; relation2"
    - Complete Paths: "entity1 ; relation1 ; entity2 ; relation2 ; entity3_1 | entity1 ; relation1 ; entity2 ; relation2 ; entity3_2 | ..."
    
    Ensures that for all complete paths, entity1 != entity3.
    """
    def __init__(self, all_kg, validation_dataset, test_dataset, steps, type = 'train', max_answers : int = 1):
        super().__init__(all_kg, validation_dataset, test_dataset, steps, type)

        # Organize walks into incomplete and complete paths
        self.data = self.data
        print(self.data[0])
        # cleaned_data = []
        # for incomplete, complete in self.data:
        #     splitted_complete = complete.split('|')
        #     if len(splitted_complete) > max_answers:
        #         continue
        #     cleaned_data.append((incomplete, complete))
        # print(f"Left with {len(cleaned_data)} / {len(self.data)} Walks of lengths {max_answers}")

        # self.data = cleaned_data


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
            for evidence in entry:
                tuple_evidence = tuple(evidence)
                walks.add(tuple_evidence)
        return list(walks)
    # def _organize_walks(self, walks):
    #     """
    #     Organize walks into a dictionary where each key is an incomplete path
    #     and the value is a list of possible complete paths.

    #     Incomplete Path: "entity1 ; relation1 ; relation2"
    #     Complete Paths: "entity1 ; relation1 ; entity2 ; relation2 ; entity3"

    #     Ensures that entity1 != entity3.

    #     Parameters:
    #     - walks (List[Tuple]): List of walks as tuples.

    #     Returns:
    #     - List of tuples: [(incomplete_path, completions), ...]
    #     """
    #     print(walks[0])
    #     data = []
    #     for walk in walks:
    #         if len(walk) != 5:
    #             continue  # Ensure the walk is exactly 2 hops: e1 ; r1 ; e2 ; r2 ; e3
    #         e1, r1, e2, r2, e3 = walk
    #         if e1 == e3:
    #             continue  # Exclude walks where e1 == e3

    #         # Define the incomplete path as "e1 ; r1 ; r2"
    #         incomplete = f"{e1} ; {r1} ; {r2}"
    #         # Define the complete path as "e1 ; r1 ; e2 ; r2 ; e3"
    #         complete = f"{e1} ; {r1} ; {e2} ; {r2} ; {e3}"
    #         data.append((incomplete, complete))
        
    #     # Convert to list of tuples (incomplete, completions)
        
        
    #     return data