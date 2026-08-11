import matplotlib.pyplot as plt

from environment import ShipEnvironment


def display_environment(environment: ShipEnvironment) -> None:
    fig, ax = plt.subplots()

    # Draw compartments
    for compartment in environment.compartments:
        polygon = compartment.geometry
        x, y = polygon.exterior.xy

        ax.fill(
            x,
            y,
            alpha=0.3,
            edgecolor="black",
        )

        centre = polygon.centroid

        ax.text(
            centre.x,
            centre.y,
            compartment.name,
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Draw walls
    for wall in environment.walls:
        line = wall.geometry
        x, y = line.xy

        ax.plot(
            x,
            y,
            color="black",
            linewidth=3,
        )

    ax.set_title("Ship Layout")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_aspect("equal")
    ax.grid(True)

    plt.show()


environment = ShipEnvironment.from_config("layouts/simple_layout2.yaml")
display_environment(environment)