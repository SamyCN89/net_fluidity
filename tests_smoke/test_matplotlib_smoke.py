import matplotlib


def test_minimal_plot_backend_agg():
    # Use non-interactive backend to avoid GUI requirements
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("smoke")
    fig.canvas.draw()  # render without saving

