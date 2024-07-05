import matplotlib.pyplot as plt
import networkx as nx

# Define tasks with duration and resource requirements for two resources
tasks = {
    'A': {'duration': 3, 'resources': [1, 0]},
    'B': {'duration': 2, 'resources': [2, 1]},
    'C': {'duration': 4, 'resources': [1, 1]},
    'D': {'duration': 1, 'resources': [0, 1]}
}

# Define resource capacities
resource_capacities = [3, 2]

# Define dependencies with maximal time lags
# (predecessor, successor, max_lag)
dependencies = [
    ('A', 'B', 5),
    ('A', 'C', 8),
    ('B', 'D', 3),
    ('C', 'D', 6)
]

# Create directed graph
G = nx.DiGraph()

# Add nodes
for task, data in tasks.items():
    G.add_node(task, duration=data['duration'], resources=data['resources'])

# Add edges with maximal time lags
for pred, succ, max_lag in dependencies:
    G.add_edge(pred, succ, max_lag=max_lag)

# Position the nodes using a layout
pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)

# Draw node labels with duration and resource requirements
node_labels = {task: f'{task}\n({data["duration"]}d, r1:{data["resources"][0]}, r2:{data["resources"][1]})' for task, data in tasks.items()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

# Draw edge labels for maximal time lags
edge_labels = {(pred, succ): f'{data["max_lag"]}d' for pred, succ, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title('RCPSP/max Instance')
plt.show()

# Sample schedule (start times for each task)
schedule = {
    'A': 0,
    'B': 4,
    'C': 3,
    'D': 7
}

# Plot the schedule
plt.figure(figsize=(10, 6))

# Plot each task as a bar
for task, start_time in schedule.items():
    duration = tasks[task]['duration']
    plt.barh(task, duration, left=start_time, color='skyblue')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Tasks')
plt.title('RCPSP/max Schedule')
plt.grid(axis='x')
plt.show()
