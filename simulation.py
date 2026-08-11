import matplotlib.pyplot as plt
from network import Network, AccessPoint, Receiver
from environment import ShipEnvironment
from random import Random


def display_network(network: Network) -> None:
    environment = network.environment
    fig, ax = plt.subplots()

    # Draw compartments
    for compartment in environment.compartments:
        polygon = compartment.geometry
        x, y = polygon.exterior.xy

        ax.fill(x, y, alpha=0.3, edgecolor="black")

        centre = polygon.centroid
        ax.text(
            centre.x, centre.y, compartment.name, 
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Draw walls
    for wall in environment.walls:
        line = wall.geometry
        x, y = line.xy

        ax.plot(x, y, color="black", linewidth=3)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw access points
    for ap in network.access_points:
        ax.scatter(*ap.xy, marker="*", s=250, color="crimson", zorder=5)
        ax.annotate(
            ap.name, ap.xy,
            xytext=(0, 8), textcoords="offset points",
            horizontalalignment="center", fontsize=8,
        )

    # Draw receivers
    for receiver in network.receivers:
        ax.scatter(*receiver.xy, marker="o", color="royalblue", zorder=5)
        ax.annotate(
            receiver.name, receiver.xy,
            xytext=(0, 8), textcoords="offset points",
            horizontalalignment="center", fontsize=8,
        )

    plt.show()

rng = Random(12345)
environment = ShipEnvironment.from_config("layouts/simple_layout2.yaml")
ap = AccessPoint("Router", "24:2f:d0:fb:85:b9", (4.5, 1.5, 1.5), 2412)
re1 = Receiver("Phone-Desk", (3, 5.9, 0.7))
re2 = Receiver("Phone-Engine", (2, 1, 1))
network = Network.from_config("layouts/simple_layout2.yaml", access_points=[ap], receivers=[re1, re2])

print("Phone-Desk to Router RSSI")
for i in range(10):
    print(network.rssi(ap, re1, rng))
print("\nPhone-Engine to Router RSSI")
for i in range(10):
    print(network.rssi(ap, re2, rng))

display_network(network)