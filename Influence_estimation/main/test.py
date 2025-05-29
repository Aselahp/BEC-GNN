import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.DiffusionModel as dm

# Create a networkx graph
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

# Create an LTM model
model = dm.LTModel(G)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter('fraction_infected', 0.1)

# Set the initial infected nodes
infected_nodes = [0]
config.add_model_initial_configuration("Infected", infected_nodes)

# Set the thresholds for each node
thresholds = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
for node, threshold in thresholds.items():
    config.add_node_configuration("threshold", node, threshold)

# Assign the configuration to the model
model.set_initial_status(config)

# Simulation execution
iterations = model.iteration_bunch(10)

# Get the final status of nodes
final_status = model.get_status()

# Print the final status
print("Final status:", final_status)