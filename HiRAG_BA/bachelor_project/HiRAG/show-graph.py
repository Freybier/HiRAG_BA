# PROJECT ADDED relative to HiRAG 67b2dfa; this module is absent upstream.
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

LABEL_TOP_N = 25  # label the N highest-degree nodes; set to None to label all

directory = "hirag_index_samples_randomness_test"

g = nx.read_graphml(f"{directory}/graph_chunk_entity_relation.graphml")
fig, ax = plt.subplots(figsize=(24, 18))
pos = nx.spring_layout(g, k=0.4, seed=42)
deg = dict(g.degree())

nx.draw(
    g,
    pos,
    node_size=[v * 10 for v in deg.values()],
    with_labels=False,
    alpha=0.7,
    width=0.3,
    ax=ax,
)

# Label only the top-degree nodes to keep it readable
if LABEL_TOP_N is None:
    label_nodes = {n: n for n in g.nodes()}
else:
    top = sorted(deg, key=deg.get, reverse=True)[:LABEL_TOP_N]
    label_nodes = {n: n for n in top}

nx.draw_networkx_labels(
    g,
    pos,
    labels=label_nodes,
    font_size=7,
    font_color="black",
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
    ax=ax,
)

out_name = directory.rstrip("/").replace("/", "_")
plt.savefig(f"graph_{out_name}.png", dpi=300, bbox_inches="tight")
plt.show()
net = Network(height="900px", width="100%", notebook=False)
net.from_nx(g)
# pyvis 0.3.2 bug: net.show() forces notebook=True, but the template only
# exists when notebook=False -> use write_html directly instead.
net.write_html(f"graph_{out_name}.html", notebook=False)  # hover/zoom/drag in browser
