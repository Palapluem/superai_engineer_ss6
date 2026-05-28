"""Indoor coordinate localization simulation package.

This package simulates a moving Arduino Nano + Modulino Movement + Modulino
Distance unit in a known room, then estimates the subject coordinate from fake
sensor packets.
"""

__all__ = [
    "room",
    "simulator",
    "estimator",
    "gait",
    "dataset_formatter",
    "io_packets",
    "plot_results",
    "report",
    "risk",
]
