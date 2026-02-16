# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}  # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)  # Count of trips for each stop
fare_rules = {}  # Mapping of route IDs to fare information
merged_fare_df = None  # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    trip_to_route = dict(zip(df_trips['trip_id'], df_trips['route_id']))

    # Map route_id to a list of stops in order of their sequence
    df_merged = df_stop_times.merge(df_trips[['trip_id', 'route_id']], on='trip_id')

    for route_id in df_merged['route_id'].unique():
        route_data = df_merged[df_merged['route_id'] == route_id]
        # Sort by stop_sequence to maintain order
        route_data = route_data.sort_values('stop_sequence')
        stops = route_data['stop_id'].unique().tolist()
        route_to_stops[route_id] = stops

    # Count trips per stop
    for trip_id in df_stop_times['trip_id'].unique():
        stops_in_trip = df_stop_times[df_stop_times['trip_id'] == trip_id]['stop_id'].values
        for stop_id in stops_in_trip:
            stop_trip_count[stop_id] += 1

    # Create fare rules for routes
    fare_rules = dict(zip(df_fare_rules['route_id'], df_fare_rules['fare_id']))

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = df_fare_rules.merge(df_fare_attributes, on='fare_id')

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.
    Returns:
        list: A list of tuples, where each tuple contains:
            - route_id (int): The ID of the route.
            - trip_count (int): The number of trips for that route.
    """
    route_trip_count = df_trips['route_id'].value_counts()
    top_5 = route_trip_count.head(5)
    return [(int(route_id), int(count)) for route_id, count in top_5.items()]

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.
    Returns:
        list: A list of tuples, where each tuple contains:
            - stop_id (int): The ID of the stop.
            - trip_count (int): The number of trips for that stop.
    """
    top_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]
    return [(int(stop_id), int(count)) for stop_id, count in top_stops]

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.
    Returns:
        list: A list of tuples, where each tuple contains:
            - stop_id (int): The ID of the stop.
            - route_count (int): The number of routes passing through that stop.
    """
    stop_route_count = defaultdict(set)
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            stop_route_count[stop_id].add(route_id)

    stop_route_freq = [(stop_id, len(routes)) for stop_id, routes in stop_route_count.items()]
    stop_route_freq.sort(key=lambda x: x[1], reverse=True)
    top_5 = stop_route_freq[:5]
    return [(int(stop_id), int(count)) for stop_id, count in top_5]

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route.
    The pairs are sorted by the combined frequency of trips passing through both stops.
    Returns:
        list: A list of tuples, where each tuple contains:
            - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
            - route_id (int): The ID of the route connecting the two stops.
    """
    stop_pair_routes = defaultdict(set)

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            pair = (stops[i], stops[i + 1])
            stop_pair_routes[pair].add(route_id)

    # Find pairs with exactly one route
    unique_pairs = []
    for pair, routes in stop_pair_routes.items():
        if len(routes) == 1:
            route_id = list(routes)[0]
            combined_freq = stop_trip_count[pair[0]] + stop_trip_count[pair[1]]
            unique_pairs.append((pair, route_id, combined_freq))

    # Sort by combined frequency
    unique_pairs.sort(key=lambda x: x[2], reverse=True)
    top_5 = unique_pairs[:5]
    return [((int(pair[0]), int(pair[1])), int(route_id)) for pair, route_id, _ in top_5]

# Function to get merged fare DataFrame
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.
    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.
    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.
    Returns:
        None
    """
    G = nx.DiGraph()

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route=route_id)

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        adjacencies = list(G.adj[node])
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'Stop ID: {node}<br># of connections: {len(adjacencies)}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Stop-Route Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.
    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.
    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    direct_routes = []
    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            start_idx = stops.index(start_stop)
            end_idx = stops.index(end_stop)
            if start_idx < end_idx:
                direct_routes.append(route_id)
    return sorted([int(r) for r in direct_routes])

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2, S, S1, S2, S3, S4')

def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.
    Returns:
        None
    """
    pyDatalog.clear()
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")

    # Define DirectRoute predicate
    DirectRoute(X, Y, R) <= (RouteHasStop(R, X, S1) & RouteHasStop(R, Y, S2) & (S1 < S2))

    # Define OptimalRoute for transfers
    OptimalRoute(X, Y, Z, R1, R2) <= (
        RouteHasStop(R1, X, S1) &
        RouteHasStop(R1, Z, S2) &
        RouteHasStop(R2, Z, S3) &
        RouteHasStop(R2, Y, S4) &
        (S1 < S2) &
        (S3 < S4) &
        (R1 != R2)
    )

    # Add route data
    add_route_data(route_to_stops)

# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.
    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.
    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for seq, stop_id in enumerate(stops):
            +RouteHasStop(route_id, stop_id, seq)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.
    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.
    Returns:
        list: A sorted list of route IDs (int) connecting the two stops.
    """
    result = DirectRoute(start, end, R)
    routes = [int(r[0]) for r in result]
    return sorted(list(set(routes)))

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.
    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.
    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria.
    """
    if max_transfers != 1:
        return []

    result = OptimalRoute(start_stop_id, end_stop_id, stop_id_to_include, R1, R2)
    paths = [(int(r1), int(via), int(r2)) for r1, r2, via in result]
    return sorted(list(set(paths)))

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.
    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.
    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria.
    """
    if max_transfers != 1:
        return []

    result = OptimalRoute(start_stop_id, end_stop_id, stop_id_to_include, R1, R2)
    paths = [(int(r1), int(via), int(r2)) for r1, r2, via in result]
    return sorted(list(set(paths)))

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.
    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.
    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria.
    """
    if max_transfers != 1:
        return []

    result = OptimalRoute(start_stop_id, end_stop_id, stop_id_to_include, R1, R2)
    paths = [(int(r1), int(via), int(r2)) for r1, r2, via in result]
    return sorted(list(set(paths)))

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.
    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.
    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    return merged_fare_df[merged_fare_df['price'] <= initial_fare]

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.
    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.
    Returns:
        dict: A summary of routes with min_price and stops.
    """
    route_summary = {}
    for route_id in pruned_df['route_id'].unique():
        route_data = pruned_df[pruned_df['route_id'] == route_id]
        min_price = route_data['price'].min()
        stops = set(route_to_stops[route_id])
        route_summary[route_id] = {
            'min_price': min_price,
            'stops': stops
        }
    return route_summary

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.
    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).
    Returns:
        list: A list representing the optimal route.
    """
    queue = deque([(start_stop_id, [], 0, 0)])  # (current_stop, path, cost, transfers)
    visited = set()

    while queue:
        current_stop, path, cost, transfers = queue.popleft()

        if current_stop == end_stop_id:
            return path

        if transfers > max_transfers:
            continue

        state = (current_stop, transfers)
        if state in visited:
            continue
        visited.add(state)

        for route_id, info in route_summary.items():
            if current_stop in info['stops']:
                route_cost = info['min_price']
                if cost + route_cost <= initial_fare:
                    stops = route_to_stops[route_id]
                    if current_stop in stops:
                        current_idx = stops.index(current_stop)
                        for next_stop in stops[current_idx + 1:]:
                            new_path = path + [(int(route_id), int(next_stop))]
                            new_transfers = transfers if (not path or path[-1][0] == route_id) else transfers + 1
                            queue.append((next_stop, new_path, cost + route_cost, new_transfers))

    return []
