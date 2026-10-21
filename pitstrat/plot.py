"""Chart of one-stop race time against pit lap for each compound pair."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed categorical order, validated for colour-blind separation.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"


def plot_one_stop(result, race, path):
    """Line per compound pair: gap to the fastest strategy for each pit lap.

    Returns False without drawing when no one-stop strategy fits the stint limits.
    """
    one_stop = result[result["Stops"] == 1].copy()
    if one_stop.empty:
        return False
    one_stop["PitLap"] = one_stop["PitLaps"].astype(int)
    fastest = result["RaceTime"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    pairs = sorted(one_stop["Strategy"].unique())
    for color, pair in zip(SERIES, pairs):
        line = one_stop[one_stop["Strategy"] == pair].sort_values("PitLap")
        gap = line["RaceTime"] - fastest
        # The legend names each pair's best pit lap, so colour is not the only cue.
        best = line.loc[gap.idxmin()]
        label = f"{pair}: best lap {best['PitLap']}, +{gap.min():.1f}s"
        ax.plot(line["PitLap"], gap, color=color, linewidth=2, label=label)
        ax.plot(best["PitLap"], gap.min(), "o", markersize=8, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2)

    ax.set_title(f"{race.circuit}: one-stop strategies by pit lap", loc="left",
                 fontsize=13, color=TEXT)
    ax.set_xlabel("Pit lap", color=MUTED)
    ax.set_ylabel("Gap to fastest strategy (s)", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), labelcolor=TEXT)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True
