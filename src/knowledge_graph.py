import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


#Need to use a undirected Graph here otherwise we will not learn the paths neccessarry for the questions
# Example we have a question: which person wrote the films directed by [Yuriy Norshteyn]	Sergei Kozlov

#But in the Graph it is:
# Film --> directed by --> Yuriy Norshteyn
# Film --> written_by --> Sergei Kozlov
def create_knowledge_graph_metaqa(data : pd.DataFrame, iterations = 0):
    index = 0
    G = nx.MultiDiGraph()
    for idx, row in tqdm(data.iterrows(), desc="Creating Knowledge Graph...", total=len(data) if iterations == 0 else iterations):
        if index > iterations:
            return G
        head = row['entity1'].strip().lower()
        relation = row['relation'].strip().lower()
        tail = row['entitiy2'].strip().lower()
        if relation == "has_tags":
            continue
        G.add_edge(head, tail, key=relation, relation=relation)
        if head != tail:
            G.add_edge(tail, head, key=f"{relation}_reversed", relation=f"{relation}_reversed")

        if iterations > 0:
            index += 1
            
    return G
def create_knowledge_graph_wikimultihop(data, iterations = 0):
    index = 0
    G = nx.MultiDiGraph()
    for entry in data['evidences']:
        if index > iterations:
            return G
        for triples in entry:
            head = triples[0].strip()
            relation = triples[1].strip()
            tail = triples[2].strip()
            G.add_edge(head, tail, key=relation, relation=relation)
        if iterations > 0:
            index += 1
            
    return G
from collections import defaultdict

def visualize_knowledge_graph(kg):
    # Generate positions for all nodes
    pos = nx.spring_layout(kg)

    # Draw the nodes and edges
    nx.draw(
        kg, pos,
        with_labels=True,
        node_size=1500,
        node_color="lightblue",
        font_size=7,
        font_weight="bold",
        edge_color="gray"
    )

    # For a MultiGraph/MultiDiGraph, multiple edges may exist between the same nodes.
    # We need to combine their 'relation' attributes into a single label.
    edge_relations = defaultdict(list)
    for u, v, data in kg.edges(data=True):
        # Extract the relation attribute (if it exists)
        relation = data.get('relation', '')
        edge_relations[(u, v)].append(relation)

    # Combine multiple relations into a single label per node pair
    # You could separate them by commas, newlines, or any other delimiter
    edge_labels = {edge: "\n".join(rels) for edge, rels in edge_relations.items()}

    # Draw the combined edge labels
    nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_color='red', font_size=7)

    # Show the plot
    plt.show()

    
def print_graph(kg):
    print("Nodes:", kg.nodes())
    print("Edges:", kg.edges(data=True))
    print(f"#Nodes: {len(kg.nodes())}")
    print(f"#Edges: {len(kg.edges(data=True))}")

#TODO CHECK IF THIS MAKES SENSE
#TODO CHECK FOR MORE relations that can be found in the questions and adjust the mapping accordingly

#################################################
#This is for MetaQA to create files with the evidences we only have start entity and answer in the files
#We need to create the evidence series entity --> relation --> entity --> relation --> entity ourself
import networkx as nx
import re
import pandas as pd
from collections import Counter
import itertools
from tqdm import tqdm

def extract_starting_entity(question):
    """
    Extracts the starting entity from the question.
    Assumes the entity is enclosed within square brackets [ ].
    """
    match = re.search(r'\[(.*?)\]', question)
    if match:
        return match.group(1).strip().lower()
    else:
        return None

def split_answers(answer_str):
    """
    Splits the answer string by '|' and trims whitespace.
    """
    return [ans.strip().lower() for ans in answer_str.split('|')]

def find_paths_with_relations(G, start, end, n=None):
    """
    Finds all paths from 'start' to 'end', optionally with exactly 'n' relations.
    Allows self-visits (revisiting nodes) in paths.
    For each path, records the sequence of relations.

    Args:
        G (networkx.Graph): The knowledge graph.
        start (str): The starting entity.
        end (str): The target entity (answer).
        n (int, optional): The exact number of relations (hops) in the path.
            If None, retrieves all paths up to a reasonable maximum number of relations.

    Returns:
        List of tuples: Each tuple contains (path, relation_sequence)
    """
    paths_rel = []

    def all_paths_allowing_cycles(G, source, target, max_depth):
        """
        Generator that yields all paths from source to target allowing cycles,
        up to a maximum path length.

        Args:
            G (networkx.Graph): The graph.
            source (str): Starting node.
            target (str): Target node.
            max_depth (int): Maximum number of nodes in the path.

        Yields:
            List of nodes representing a path from source to target.
        """
        stack = [(source, [source])]
        while stack:
            (vertex, path) = stack.pop()
            if len(path) > max_depth:
                continue
            for neighbor in G.neighbors(vertex):
                new_path = path + [neighbor]
                if neighbor == target:
                    yield new_path
                if len(new_path) < max_depth:
                    stack.append((neighbor, new_path))

    try:
        if n is not None:
            # Exact number of relations: n hops => n+1 nodes
            max_depth = n + 1
            all_paths = list(all_paths_allowing_cycles(G, source=start, target=end, max_depth=max_depth))
            # Filter paths that have exactly n relations (i.e., n+1 nodes)
            exact_paths = [path for path in all_paths if len(path) == n + 1]
            # Iterate through exact paths
            for path in exact_paths:
                relations_per_pair = []
                for i in range(len(path) - 1):
                    node1 = path[i]
                    node2 = path[i + 1]
                    rel = []
                    if G.has_edge(node1, node2):
                        edge_attrs = G.get_edge_data(node1, node2)
                        if G.is_multigraph():
                            # Collect all unique relations, normalize them
                            rel = list(set(
                                attrs.get('relation').strip().lower() for key, attrs in edge_attrs.items() if 'relation' in attrs
                            ))
                        else:
                            rel_attr = edge_attrs.get('relation')
                            if rel_attr:
                                rel = [rel_attr.strip().lower()]
                    if not rel:
                        rel = ['unknown_relation']
                    relations_per_pair.append(rel)

                # Generate all combinations of relations
                for relation_combo in itertools.product(*relations_per_pair):
                    #print(f"  Generated path: {path} with relations {relation_combo}")
                    paths_rel.append((path, relation_combo))
        else:
            # Retrieve all paths allowing self-visits up to a reasonable max_depth
            # Define max_depth based on graph size to prevent excessive computation
            # For example, twice the number of nodes
            num_nodes = G.number_of_nodes()
            max_depth = num_nodes * 2  # Adjust as needed
            all_paths = list(all_paths_allowing_cycles(G, source=start, target=end, max_depth=max_depth))
            print(f"  Found {len(all_paths)} paths from '{start}' to '{end}':")
            for path in all_paths:
                relations_per_pair = []
                for i in range(len(path) - 1):
                    node1 = path[i]
                    node2 = path[i + 1]
                    rel = []
                    if G.has_edge(node1, node2):
                        edge_attrs = G.get_edge_data(node1, node2)
                        if G.is_multigraph():
                            # Collect all unique relations, normalize them
                            rel = list(set(
                                attrs.get('relation').strip().lower() for key, attrs in edge_attrs.items() if 'relation' in attrs
                            ))
                        else:
                            rel_attr = edge_attrs.get('relation')
                            if rel_attr:
                                rel = [rel_attr.strip().lower()]
                    if not rel:
                        rel = ['unknown_relation']
                    relations_per_pair.append(rel)

                # Generate all combinations of relations
                for relation_combo in itertools.product(*relations_per_pair):
                    paths_rel.append((path, relation_combo))
    except Exception as e:
        # Handle exceptions such as no path exists
        if isinstance(e, nx.NetworkXNoPath):
            if n is not None:
                print(f"  No path found from '{start}' to '{end}' with exactly {n} relations.")
            else:
                print(f"  No path found from '{start}' to '{end}'.")
        else:
            # For other exceptions, re-raise
            raise e
    return paths_rel

def find_common_relation_sequences(all_answers_paths, n):
    """
    Identifies relation sequences that are present in all answers' relation sequences.

    Args:
        all_answers_paths (List[List[Tuple[List[str], Tuple[str, ...]]]]): 
            A list where each element corresponds to an answer and contains a list of (path, relation_sequence) tuples.
        n (int): The exact number of relations (hops) to consider.

    Returns:
        Set of relation sequences (tuples) that are common to all answers.
    """
    # For each answer, collect all relation sequences
    relation_sequences_per_answer = []
    for answer_idx, answer_paths in enumerate(all_answers_paths):
        rel_seqs = set(
            tuple(rel.lower().strip() for rel in rel_seq)
            for path, rel_seq in answer_paths
            if len(rel_seq) == n
        )
        #print(f"  Answer {answer_idx + 1} relation sequences: {rel_seqs}")
        relation_sequences_per_answer.append(rel_seqs)

    if not relation_sequences_per_answer:
        return set()

    # Find intersection of all relation sets
    common_rel_seqs = set.intersection(*relation_sequences_per_answer)
    #print(f"  Common relation sequences across all answers: {common_rel_seqs}")

    return common_rel_seqs

def select_paths_with_specific_relation_sequence(all_answers_paths, selected_rel_seq):
    """
    Selects one path per answer that matches the specified relation sequence.

    Args:
        all_answers_paths (List[List[Tuple[List[str], Tuple[str, ...]]]]): 
            A list where each element corresponds to an answer and contains a list of (path, relation_sequence) tuples.
        selected_rel_seq (Tuple[str, ...]): 
            The specific relation sequence to match.

    Returns:
        List of selected (path, relation_sequence), one for each answer.
    """
    selected_paths = []
    for answer_idx, answer_paths in enumerate(all_answers_paths):
        selected_path = None
        selected_rel_seq_found = None
        for path, rel_seq in answer_paths:
            # Uncomment the following lines for detailed debugging
            # print(f"    Checking path for Answer {answer_idx + 1}: {path} with relations {rel_seq}")
            # print(f"      Path relation sequence type: {type(rel_seq)}")
            # print(f"      Selected relation sequence type: {type(selected_rel_seq)}")
            # print(f"      Selected relation sequence: {selected_rel_seq}")
            # print(f"      Current path's relation sequence: {rel_seq}")
            if rel_seq == selected_rel_seq:
                selected_path = path
                selected_rel_seq_found = rel_seq
                # Uncomment the following line for detailed debugging
                # print(f"        --> Match found for Answer {answer_idx + 1}")
                break  # Select the first matching path
        if selected_path and selected_rel_seq_found:
            selected_paths.append((selected_path, selected_rel_seq_found))
        else:
            # If no path contains the specific relation sequence, append None
            print(f"    No path with the specific relation sequence found for Answer {answer_idx + 1}.")
            selected_paths.append((None, None))
    return selected_paths

def process_questions(file_path, G, n):
    """
    Processes each question-answer pair, finds the corresponding paths in the KG,
    ensures relations are consistent across multiple answers with exactly 'n' hops,
    and returns a DataFrame with the results.

    Args:
        file_path (str): Path to the TXT file containing questions and answers.
        G (networkx.Graph): The knowledge graph.
        n (int): The exact number of relations (hops) to consider.

    Returns:
        pandas.DataFrame: DataFrame containing the question, answers, and evidences.
    """
    # Define the phrase-to-relation mapping
    PHRASE_TO_RELATION = {
        "written by": "written_by_reversed",
        "directed by": "directed_by_reversed",
        "acted by": "acted_by_reversed",
        "starred by": "starred_actors_reversed",
    }

    # Compile regex patterns for phrases
    phrase_patterns = {phrase: re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE) 
                       for phrase in PHRASE_TO_RELATION.keys()}

    # Define priority order for phrases (optional)
    PHRASE_PRIORITY = ["written by", "directed by", "acted by", "starred by"]

    # Read the file into a pandas DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None, names=['question', 'answer'])

    # Lists to store DataFrame columns
    questions = []
    answers_list = []
    evidences_list = []

    # Iterate over each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Questions"):
        question = row['question']
        answer_str = row['answer'].lower()

        # Extract the starting entity from the question
        start_entity = extract_starting_entity(question)
        if not start_entity:
            print(f"  No starting entity found in question: '{question}'. Skipping.\n")
            continue  # Skip to the next row

        # Split the answers by '|'
        answers = split_answers(answer_str)

        # Collect all paths for each answer
        all_answers_paths = []
        missing_paths = False  # Flag to check if any answer has no path
        for answer in answers:
            paths_rel = find_paths_with_relations(G, start_entity, answer, n)
            if not paths_rel:
                print(f"    No path found from '{start_entity}' to '{answer}'.")
                missing_paths = True
            all_answers_paths.append(paths_rel)

        if missing_paths:
            print(f"  Some answers have no paths. Skipping this question.\n")
            continue  # Skip processing if any answer lacks a path

        # If there are multiple answers, ensure relation sequences are common across all
        if len(answers) > 1:
            common_rel_seqs = find_common_relation_sequences(all_answers_paths, n)
            if not common_rel_seqs:
                # No common relation sequences found across all answers
                print(f"  No common relation sequences found across all answers for question: '{question}'. Skipping.\n")
                continue  # Skip if no common relation sequences

            # Initialize selected_rel_seq to None
            selected_rel_seq = None

            # Check for specific phrases in the question to prioritize relation selection
            for phrase in PHRASE_PRIORITY:
                relation = PHRASE_TO_RELATION[phrase]
                if phrase_patterns[phrase].search(question):
                    # Find a relation sequence that includes the corresponding relation
                    matching_seqs = [seq for seq in common_rel_seqs if relation in seq]
                    if matching_seqs:
                        selected_rel_seq = matching_seqs[0]  # Select the first matching sequence
                        #print(f"  Phrase '{phrase}' found in question. Selected relation sequence: {selected_rel_seq}")
                        break  # Stop after finding the first matching phrase

            # If no specific phrase is found, select the first common relation sequence
            if not selected_rel_seq:
                selected_rel_seq = next(iter(common_rel_seqs))
                #print(f"  No specific phrase found in question. Selected first common relation sequence: {selected_rel_seq}")

            # Select paths that match this specific relation sequence
            selected_paths = select_paths_with_specific_relation_sequence(all_answers_paths, selected_rel_seq)

            # Check if all selected paths are valid (not None)
            if any(path is None for path, rel_seq in selected_paths):
                print(f"  Not all answers have paths with the selected common relation sequence for question: '{question}'. Skipping.\n")
                continue  # Skip if any answer does not have a path with the common relation sequence

            # Prepare evidences for the current question
            evidences = []
            for path, rel_seq in selected_paths:
                # Construct the path string with relations
                path_with_relations = []
                for j in range(len(path)):
                    path_with_relations.append(str(path[j]))
                    if j < len(rel_seq):
                        path_with_relations.append(str(rel_seq[j]))
                # Store as tuple
                evidences.append(tuple(path_with_relations))

            # Append data to lists
            questions.append(question)
            answers_list.append(answers)
            evidences_list.append(evidences)
        else:
            # For single answers, select the first available path
            if all_answers_paths[0]:
                selected_path, selected_rel_seq = all_answers_paths[0][0]  # First path tuple
                # Construct the path string with relations
                path_with_relations = []
                for j in range(len(selected_path)):
                    path_with_relations.append(str(selected_path[j]))
                    if j < len(selected_rel_seq):
                        path_with_relations.append(str(selected_rel_seq[j]))
                # Store as tuple
                evidences = [tuple(path_with_relations)]

                # Append data to lists
                questions.append(question)
                answers_list.append(answers)
                evidences_list.append(evidences)
            else:
                print(f"    No path selected for answer: '{answers[0]}'.\n")
                evidences_list.append([None])

    # Create the DataFrame
    results_df = pd.DataFrame({
        'question': questions,
        'answers': answers_list,
        'evidences': evidences_list
    })

    return results_df

