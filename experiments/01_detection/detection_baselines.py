"""The four detectors in the current Group 01 experiment."""
from __future__ import annotations

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.method_names import RACA_HIM
from .proposed_detection import DetectionResult, detect_proposed
from .iedi_adapted import detect_iedi
from .ctcv_adapted import detect_ctcv
from .dynaedi_adapted import detect_dynaedi

METHODS = [RACA_HIM, "I-EDI-Adapted", "CTCV-Adapted", "DynaEDI+-Adapted"]
IMPLEMENTED_METHODS = list(METHODS)
METHOD_FUNCTIONS = {
    RACA_HIM: detect_proposed,
    "I-EDI-Adapted": detect_iedi,
    "CTCV-Adapted": detect_ctcv,
    "DynaEDI+-Adapted": detect_dynaedi,
}


def detect_methods(data: AlgorithmInput, config: DetectionConfig) -> list[DetectionResult]:
    """Execute all four methods in their declared order."""
    return [METHOD_FUNCTIONS[method](data, config) for method in METHODS]
