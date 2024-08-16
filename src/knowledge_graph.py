import networkx as nx
import matplotlib.pyplot as plt

def create_knowledge_graph(data, iterations = 0):
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
def visualize_knowledge_graph(kg):
    # Visualize the graph
    pos = nx.spring_layout(kg)  # Positions for all nodes

    # Draw nodes and edges
    nx.draw(kg, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=7, font_weight="bold", edge_color="gray")

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(kg, 'relation')
    nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels, font_color='red', font_size=7)

    # Show the plot
    plt.show()
    
def print_graph(kg):
    print("Nodes:", kg.nodes())
    print("Edges:", kg.edges(data=True))
    print(f"#Nodes: {len(kg.nodes())}")
    print(f"#Edges: {len(kg.edges(data=True))}")