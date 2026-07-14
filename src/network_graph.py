import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Load social data
df = pd.read_csv("data/telegram_posts_labelled.csv")
df["date"] = pd.to_datetime(df["date"], utc=True)

print(f"Total posts: {len(df)}")
print(f"Channels: {df['channel'].nunique()}")
print(f"Channel list: {df['channel'].unique().tolist()}")

# Build bipartite graph: channels → messages
G = nx.Graph()

# Add channel nodes
for channel in df["channel"].unique():
    G.add_node(channel, node_type="channel",
               post_count=len(df[df["channel"]==channel]))

# Detect near-duplicate messages across channels
# (same/similar text posted in multiple channels = coordinated activity)
print("\n=== COORDINATED POSTING ANALYSIS ===")
text_to_channels = defaultdict(list)

for _, row in df.iterrows():
    # Use first 50 chars as fingerprint
    fingerprint = str(row["text"])[:50].strip().lower()
    text_to_channels[fingerprint].append({
        "channel": row["channel"],
        "date": row["date"],
        "label": row["label"]
    })

# Find messages posted in multiple channels
coordinated = {k: v for k, v in text_to_channels.items() if len(v) > 1}
print(f"Near-duplicate messages across channels: {len(coordinated)}")

# Add coordination edges
coord_edge_count = 0
for fingerprint, posts in coordinated.items():
    channels_involved = list(set([p["channel"] for p in posts]))
    if len(channels_involved) > 1:
        for i in range(len(channels_involved)):
            for j in range(i+1, len(channels_involved)):
                G.add_edge(channels_involved[i], channels_involved[j],
                          weight=1,
                          edge_type="coordinated_message")
                coord_edge_count += 1

print(f"Coordination edges added: {coord_edge_count}")

# Channel similarity edges based on label distribution
print("\n=== CHANNEL SIMILARITY NETWORK ===")
channel_profiles = {}
for channel in df["channel"].unique():
    ch_df = df[df["channel"] == channel]
    profile = {
        "suspicious_ratio": (ch_df["label"] == 2).mean(),
        "tip_ratio": (ch_df["label"] == 1).mean(),
        "neutral_ratio": (ch_df["label"] == 0).mean(),
        "post_count": len(ch_df)
    }
    channel_profiles[channel] = profile
    print(f"\n{channel}:")
    print(f"  Posts: {profile['post_count']}")
    print(f"  Suspicious: {profile['suspicious_ratio']*100:.0f}%")
    print(f"  Open Tips: {profile['tip_ratio']*100:.0f}%")
    print(f"  Neutral: {profile['neutral_ratio']*100:.0f}%")

# Graph metrics
print("\n=== NETWORK METRICS ===")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

if G.number_of_nodes() > 0:
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    print("\nDegree centrality (higher = more connected/influential):")
    for node, cent in sorted(degree_cent.items(), key=lambda x: -x[1]):
        print(f"  {node}: {cent:.3f}")

    # Detect communities
    if G.number_of_edges() > 0:
        communities = list(nx.community.greedy_modularity_communities(G))
        print(f"\nCommunities detected: {len(communities)}")
        for i, comm in enumerate(communities):
            print(f"  Community {i+1}: {list(comm)}")

# Visualize the network
print("\nGenerating network visualization...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Channel network
ax1 = axes[0]
pos = nx.spring_layout(G, seed=42) if G.number_of_nodes() > 0 else {}

node_colors = []
node_sizes = []
for node in G.nodes():
    profile = channel_profiles.get(node, {})
    susp = profile.get("suspicious_ratio", 0)
    if susp > 0.7:
        node_colors.append("#FF4B4B")
    elif susp > 0.4:
        node_colors.append("#FFA500")
    else:
        node_colors.append("#534AB7")
    node_sizes.append(300 + profile.get("post_count", 10) * 10)

nx.draw_networkx(G, pos, ax=ax1,
                node_color=node_colors,
                node_size=node_sizes,
                font_size=7,
                font_color="white",
                edge_color="gray",
                alpha=0.9)
ax1.set_title("Channel Coordination Network\n(Red=High Suspicious, Blue=Low Suspicious)",
              fontsize=11)
ax1.axis("off")

# Plot 2: Suspicious ratio bar chart per channel
ax2 = axes[1]
channels = list(channel_profiles.keys())
susp_ratios = [channel_profiles[c]["suspicious_ratio"] for c in channels]
colors = ["#FF4B4B" if s > 0.7 else "#FFA500" if s > 0.4 else "#534AB7"
          for s in susp_ratios]

bars = ax2.barh([c.replace("_", "\n") for c in channels],
                susp_ratios, color=colors)
ax2.set_xlabel("Suspicious Post Ratio")
ax2.set_title("Suspicious Content Ratio by Channel", fontsize=11)
ax2.set_xlim(0, 1)
for bar, val in zip(bars, susp_ratios):
    ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f"{val*100:.0f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("docs/network_graph.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: docs/network_graph.png")

# Save network data
network_data = pd.DataFrame([
    {
        "channel": ch,
        "post_count": p["post_count"],
        "suspicious_ratio": p["suspicious_ratio"],
        "tip_ratio": p["tip_ratio"],
        "neutral_ratio": p["neutral_ratio"],
        "degree_centrality": degree_cent.get(ch, 0) if G.number_of_nodes() > 0 else 0
    }
    for ch, p in channel_profiles.items()
])
network_data.to_csv("docs/network_analysis.csv", index=False)
print("Saved: docs/network_analysis.csv")
print("\nNetwork analysis complete.")