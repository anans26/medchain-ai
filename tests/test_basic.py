import pytest
from ai_model import MedicalAIInference, create_test_patients

def test_basic_diagnosis():
    model = MedicalAIInference()
    patient = create_test_patients()[0]
    result = model.diagnose_rare_disease(patient)
    assert "primary_diagnosis" in result
    assert 0 <= result["confidence"] <= 1

