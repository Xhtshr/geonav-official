import networkx as nx


structured_data = None
# maintaining a graph of city
G = nx.DiGraph()

# add Node
"""
landmark: {(x,y), label}
Target
Surrounding
"""
G.add_node(structured_data["Target"], type="target")

# add edge


