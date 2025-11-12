#!/usr/bin/env python3
"""
Simple DFC Speed Results Analysis
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def setup_logging():
    """Configure logging from YAML if available; fallback to basicConfig."""
    cfg_path = os.getenv("NET_FLUIDITY_LOGGING", "config/logging.yaml")
    try:
        if os.path.exists(cfg_path):
            from logging.config import dictConfig

            import yaml

            with open(cfg_path) as f:
                dictConfig(yaml.safe_load(f))
            return
    except Exception:
        # fall back to basic config
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


setup_logging()
logger = logging.getLogger(__name__)


def main():
    # Load the data
    results_file = Path(
        "/media/samy/Elements2/Proyectos/LauraHarsan/results/julien_caillette/speed/speed_dfc_lag=1_tau=5_wmax=100_wmin=5.npz"
    )

    logger.info("Loading DFC speed results from %s", results_file)
    data = np.load(results_file, allow_pickle=True)

    vel_data = data["vel"]
    speed_medians = data["speed_median"]

    logger.info("Number of animals: %s", len(vel_data))
    logger.info("Speed medians shape: %s", speed_medians.shape)

    # Simple statistics on speed medians
    logger.info("=== SPEED MEDIAN ANALYSIS ===")
    logger.info(
        "Overall median speed range: [%.6f, %.6f]",
        float(np.nanmin(speed_medians)),
        float(np.nanmax(speed_medians)),
    )
    logger.info("Overall median speed mean: %.6f", float(np.nanmean(speed_medians)))

    # Try to analyze individual animal speeds
    logger.info("=== INDIVIDUAL ANIMAL ANALYSIS ===")
    valid_animals = 0
    total_measurements = 0

    for i in range(len(vel_data)):
        try:
            vel = vel_data[i]
            if vel is not None:
                # Convert to array if needed
                if not isinstance(vel, np.ndarray):
                    vel = np.array(vel)

                n_measurements = len(vel.flatten())
                total_measurements += n_measurements
                valid_animals += 1

                if i < 5:  # Show first 5 animals
                    vel_flat = vel.flatten()
                    logger.debug(
                        "Animal %s: %s measurements, range [%.6f, %.6f]",
                        i,
                        n_measurements,
                        float(vel_flat.min()),
                        float(vel_flat.max()),
                    )
            else:
                logger.warning("Animal %s: No data", i)

        except Exception as e:
            logger.exception("Animal %s: Error - %s", i, e)

    logger.info(
        "Summary: %s/%s animals with valid data",
        valid_animals,
        len(vel_data),
    )
    logger.info("Total speed measurements: %s", f"{total_measurements:,}")

    # Create simple visualization
    logger.info("=== CREATING VISUALIZATION ===")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Speed medians across animals (averaged across windows and tau)
    animal_avg_speeds = np.nanmean(speed_medians, axis=(1, 2))
    axes[0, 0].bar(range(len(animal_avg_speeds)), animal_avg_speeds)
    axes[0, 0].set_title("Average Speed per Animal")
    axes[0, 0].set_xlabel("Animal ID")
    axes[0, 0].set_ylabel("Average Speed")

    # 2. Speed medians across window sizes (averaged across animals and tau)
    window_avg_speeds = np.nanmean(speed_medians, axis=(0, 2))
    axes[0, 1].plot(window_avg_speeds)
    axes[0, 1].set_title("Speed vs Window Size")
    axes[0, 1].set_xlabel("Window Index")
    axes[0, 1].set_ylabel("Average Speed")

    # 3. Speed medians across tau values (averaged across animals and windows)
    tau_avg_speeds = np.nanmean(speed_medians, axis=(0, 1))
    axes[1, 0].bar(range(len(tau_avg_speeds)), tau_avg_speeds)
    axes[1, 0].set_title("Speed vs Tau Value")
    axes[1, 0].set_xlabel("Tau Index")
    axes[1, 0].set_ylabel("Average Speed")

    # 4. Distribution of all median speeds
    all_medians = speed_medians.flatten()
    all_medians = all_medians[~np.isnan(all_medians)]
    axes[1, 1].hist(all_medians, bins=50, alpha=0.7, histtype="step")

    axes[1, 1].set_title("Distribution of Speed Medians")
    axes[1, 1].set_xlabel("Speed")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()

    # Save the plot
    output_file = "dfc_speed_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info("Visualization saved as: %s", output_file)

    # Don't show the plot in headless environment
    # plt.show()


if __name__ == "__main__":
    main()
