"""
scalepredict.monitor
====================
W-Twin online training trajectory monitor.

Main entry points:
    WTwinMonitor   — online monitor (stream step-by-step)
    PowerLawBaseline — default baseline predictor
    run_benchmark  — comparative benchmark vs Threshold / CUSUM
"""

from scalepredict.monitor.wtwin import WTwinMonitor, WTwinState
from scalepredict.monitor.baseline import PowerLawBaseline, BaseBaseline
from scalepredict.monitor.benchmark import run_benchmark
from scalepredict.monitor.suggest import suggest, Suggestion

__all__ = [
    "WTwinMonitor",
    "WTwinState",
    "PowerLawBaseline",
    "BaseBaseline",
    "run_benchmark",
    "suggest",
    "Suggestion",
]
