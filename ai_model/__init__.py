"""AI Model package for MedChain AI.

Exposes high-level interfaces so that other modules can simply do:

    from ai_model import MedicalAIInference, create_test_patients

This file purposely keeps the import lightweight; all heavy libraries are
import-guarded inside pretrained_medical_ai.py to avoid slowing down simple
package discovery or CLI operations.
"""

from .pretrained_medical_ai import MedicalAIInference, create_test_patients  # noqa: F401

