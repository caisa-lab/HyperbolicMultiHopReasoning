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