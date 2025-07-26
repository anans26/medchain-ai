#!/usr/bin/env python3
"""
Advanced Test Suite for MedChain AI
Comprehensive testing framework for model evaluation and validation
"""

import sys
import os
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

# Add the ai_model directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai_model'))

try:
    from pretrained_medical_ai import MedicalAIInference, create_test_patients
    print("✅ Successfully imported medical AI modules")
except ImportError as e:
    print(f"❌ Failed to import medical AI: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "test_results.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_test_suite")

class AdvancedTestSuite:
    """
    Comprehensive test suite for MedChain AI model evaluation
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the test suite
        
        Args:
            output_dir: Directory to save test results and visualizations
        """
        self.model = MedicalAIInference()
        
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(__file__), "results", 
                                          datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test datasets
        self.test_patients = create_test_patients()
        self.synthetic_patients = self._generate_synthetic_patients()
        self.edge_cases = self._generate_edge_cases()
        self.adversarial_cases = self._generate_adversarial_cases()
        
        # Test results
        self.results = {
            "accuracy": {},
            "performance": {},
            "robustness": {},
            "privacy": {},
            "explainability": {},
            "timestamp": datetime.now().isoformat()
        }
        
    def _generate_synthetic_patients(self, count: int = 50) -> List[Dict]:
        """Generate synthetic patients with diverse characteristics"""
        
        diseases = [
            "Huntington Disease", "Cystic Fibrosis", "Myasthenia Gravis",
            "Amyotrophic Lateral Sclerosis", "Duchenne Muscular Dystrophy",
            "Wilson Disease", "Fabry Disease", "Gaucher Disease",
            "Pompe Disease", "Tay-Sachs Disease"
        ]
        
        disease_symptoms = {
            "Huntington Disease": ["chorea", "involuntary movements", "cognitive decline", "behavioral changes", "depression", "difficulty swallowing"],
            "Cystic Fibrosis": ["chronic cough", "thick mucus", "recurrent lung infections", "poor weight gain", "salty skin", "digestive problems"],
            "Myasthenia Gravis": ["muscle weakness", "double vision", "drooping eyelids", "difficulty swallowing", "slurred speech", "fatigue"],
            "Amyotrophic Lateral Sclerosis": ["muscle weakness", "muscle atrophy", "fasciculations", "speech problems", "difficulty swallowing"],
            "Duchenne Muscular Dystrophy": ["muscle weakness", "muscle atrophy", "contractures", "scoliosis", "cardiomyopathy"],
            "Wilson Disease": ["liver problems", "neurological symptoms", "psychiatric symptoms", "tremor", "dystonia"],
            "Fabry Disease": ["pain", "burning sensation", "rash", "kidney problems", "heart problems", "hearing loss"],
            "Gaucher Disease": ["fatigue", "bone pain", "enlarged liver", "enlarged spleen", "bleeding", "bruising"],
            "Pompe Disease": ["muscle weakness", "breathing problems", "heart problems", "feeding difficulties"],
            "Tay-Sachs Disease": ["developmental delay", "seizures", "vision loss", "hearing loss", "muscle weakness"],
        }
        
        patients = []
        
        for _ in range(count):
            # Select random disease
            disease = np.random.choice(diseases)
            symptoms = disease_symptoms[disease]
            
            # Select random subset of symptoms (at least 2)
            num_symptoms = max(2, np.random.randint(1, len(symptoms) + 1))
            selected_symptoms = np.random.choice(symptoms, size=num_symptoms, replace=False).tolist()
            
            # Generate random age and gender
            age = np.random.randint(5, 85)
            gender = np.random.choice(["male", "female"])
            
            # Generate random lab values
            lab_values = {
                "hemoglobin": np.random.uniform(8.0, 18.0),
                "white_blood_cells": np.random.uniform(3.0, 15.0),
                "creatinine": np.random.uniform(0.5, 2.0),
                "glucose": np.random.uniform(70, 200),
                "alt": np.random.uniform(10, 100)
            }
            
            # Create clinical notes
            clinical_notes = f"{age}-year-old {gender} presenting with {', '.join(selected_symptoms)}. "
            clinical_notes += f"Patient reports symptoms began {np.random.randint(1, 24)} months ago."
            
            patient = {
                "symptoms": selected_symptoms,
                "clinical_notes": clinical_notes,
                "age": age,
                "gender": gender,
                "lab_values": lab_values,
                "expected_diagnosis": disease
            }
            
            patients.append(patient)
            
        return patients
    
    def _generate_edge_cases(self) -> List[Dict]:
        """Generate edge cases to test model robustness"""
        
        edge_cases = [
            # Case with minimal symptoms
            {
                "symptoms": ["fatigue"],
                "clinical_notes": "Patient reports feeling tired all the time.",
                "age": 45,
                "gender": "female",
                "lab_values": {"hemoglobin": 12.5, "white_blood_cells": 7.2},
                "expected_diagnosis": "Unknown",
                "case_type": "minimal_symptoms"
            },
            
            # Case with conflicting symptoms from multiple diseases
            {
                "symptoms": ["chorea", "thick mucus", "muscle weakness", "liver problems"],
                "clinical_notes": "Patient with mixed presentation of multiple symptom clusters.",
                "age": 35,
                "gender": "male",
                "lab_values": {"hemoglobin": 13.2, "white_blood_cells": 8.1},
                "expected_diagnosis": "Complex",
                "case_type": "conflicting_symptoms"
            },
            
            # Case with extremely abnormal lab values
            {
                "symptoms": ["fatigue", "weakness"],
                "clinical_notes": "Patient with severe laboratory abnormalities.",
                "age": 60,
                "gender": "male",
                "lab_values": {
                    "hemoglobin": 5.2,
                    "white_blood_cells": 25.0,
                    "creatinine": 4.5,
                    "glucose": 450,
                    "alt": 500
                },
                "expected_diagnosis": "Complex",
                "case_type": "abnormal_labs"
            },
            
            # Case with pediatric patient
            {
                "symptoms": ["muscle weakness", "poor weight gain"],
                "clinical_notes": "3-year-old with developmental concerns.",
                "age": 3,
                "gender": "female",
                "lab_values": {"hemoglobin": 10.5, "white_blood_cells": 9.2},
                "expected_diagnosis": "Complex",
                "case_type": "pediatric"
            },
            
            # Case with elderly patient and multiple comorbidities
            {
                "symptoms": ["fatigue", "cognitive decline", "breathing problems"],
                "clinical_notes": "92-year-old with multiple chronic conditions including diabetes, hypertension, and COPD.",
                "age": 92,
                "gender": "male",
                "lab_values": {"hemoglobin": 11.0, "white_blood_cells": 10.5},
                "expected_diagnosis": "Complex",
                "case_type": "elderly_comorbid"
            }
        ]
        
        return edge_cases
    
    def _generate_adversarial_cases(self) -> List[Dict]:
        """Generate adversarial cases to test model robustness"""
        
        adversarial_cases = [
            # Case with contradictory information
            {
                "symptoms": ["chorea", "involuntary movements"],
                "clinical_notes": "Patient denies any movement disorders or involuntary movements.",
                "age": 40,
                "gender": "female",
                "lab_values": {"hemoglobin": 13.5, "white_blood_cells": 7.8},
                "expected_diagnosis": "Uncertain",
                "case_type": "contradictory"
            },
            
            # Case with irrelevant information
            {
                "symptoms": ["muscle weakness", "double vision"],
                "clinical_notes": "Patient enjoys hiking and has traveled to 15 countries. Recently adopted a pet dog. Works as an accountant.",
                "age": 35,
                "gender": "male",
                "lab_values": {"hemoglobin": 14.2, "white_blood_cells": 6.9},
                "expected_diagnosis": "Myasthenia Gravis",
                "case_type": "irrelevant_info"
            },
            
            # Case with misspelled symptoms
            {
                "symptoms": ["muscl weekness", "dubble vission", "drooping eylids"],
                "clinical_notes": "Paitent with proggresive symtoms over 6 monthes.",
                "age": 50,
                "gender": "female",
                "lab_values": {"hemoglobin": 12.8, "white_blood_cells": 7.5},
                "expected_diagnosis": "Myasthenia Gravis",
                "case_type": "misspelled"
            },
            
            # Case with extremely long clinical notes
            {
                "symptoms": ["chronic cough", "thick mucus"],
                "clinical_notes": "Patient with respiratory symptoms. " + "Additional details. " * 100,
                "age": 25,
                "gender": "male",
                "lab_values": {"hemoglobin": 13.0, "white_blood_cells": 8.0},
                "expected_diagnosis": "Cystic Fibrosis",
                "case_type": "long_notes"
            },
            
            # Case with empty fields
            {
                "symptoms": ["chorea", "cognitive decline"],
                "clinical_notes": "",
                "age": 0,
                "gender": "",
                "lab_values": {},
                "expected_diagnosis": "Huntington Disease",
                "case_type": "empty_fields"
            }
        ]
        
        return adversarial_cases
    
    def run_accuracy_tests(self) -> Dict:
        """Run accuracy tests on the model"""
        
        logger.info("Running accuracy tests...")
        
        results = {
            "standard_cases": {},
            "synthetic_cases": {},
            "edge_cases": {},
            "adversarial_cases": {}
        }
        
        # Test standard cases
        correct_standard = 0
        for i, patient in enumerate(self.test_patients):
            logger.info(f"Testing standard case {i+1}/{len(self.test_patients)}")
            
            # Get expected diagnosis based on symptoms
            expected = self._get_expected_diagnosis(patient)
            
            # Get model prediction
            start_time = time.time()
            diagnosis = self.model.diagnose_rare_disease(patient)
            inference_time = time.time() - start_time
            
            # Check if prediction matches expected
            predicted = diagnosis["primary_diagnosis"]
            confidence = diagnosis["confidence"]
            
            is_correct = expected in predicted
            
            if is_correct:
                correct_standard += 1
                
            results["standard_cases"][i] = {
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "correct": is_correct,
                "inference_time": inference_time
            }
        
        # Calculate accuracy for standard cases
        standard_accuracy = correct_standard / len(self.test_patients) if self.test_patients else 0
        results["standard_accuracy"] = standard_accuracy
        
        # Test synthetic cases
        correct_synthetic = 0
        for i, patient in enumerate(self.synthetic_patients):
            logger.info(f"Testing synthetic case {i+1}/{len(self.synthetic_patients)}")
            
            expected = patient["expected_diagnosis"]
            
            # Get model prediction
            start_time = time.time()
            diagnosis = self.model.diagnose_rare_disease(patient)
            inference_time = time.time() - start_time
            
            # Check if prediction matches expected
            predicted = diagnosis["primary_diagnosis"]
            confidence = diagnosis["confidence"]
            
            is_correct = expected in predicted
            
            if is_correct:
                correct_synthetic += 1
                
            results["synthetic_cases"][i] = {
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "correct": is_correct,
                "inference_time": inference_time
            }
        
        # Calculate accuracy for synthetic cases
        synthetic_accuracy = correct_synthetic / len(self.synthetic_patients) if self.synthetic_patients else 0
        results["synthetic_accuracy"] = synthetic_accuracy
        
        # Test edge cases
        for i, patient in enumerate(self.edge_cases):
            logger.info(f"Testing edge case {i+1}/{len(self.edge_cases)}")
            
            expected = patient["expected_diagnosis"]
            case_type = patient["case_type"]
            
            # Get model prediction
            start_time = time.time()
            diagnosis = self.model.diagnose_rare_disease(patient)
            inference_time = time.time() - start_time
            
            # Check if prediction matches expected
            predicted = diagnosis["primary_diagnosis"]
            confidence = diagnosis["confidence"]
            
            results["edge_cases"][i] = {
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "case_type": case_type,
                "inference_time": inference_time
            }
        
        # Test adversarial cases
        for i, patient in enumerate(self.adversarial_cases):
            logger.info(f"Testing adversarial case {i+1}/{len(self.adversarial_cases)}")
            
            expected = patient["expected_diagnosis"]
            case_type = patient["case_type"]
            
            # Get model prediction
            start_time = time.time()
            diagnosis = self.model.diagnose_rare_disease(patient)
            inference_time = time.time() - start_time
            
            # Check if prediction matches expected
            predicted = diagnosis["primary_diagnosis"]
            confidence = diagnosis["confidence"]
            
            results["adversarial_cases"][i] = {
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "case_type": case_type,
                "inference_time": inference_time
            }
        
        # Calculate overall accuracy
        overall_accuracy = (correct_standard + correct_synthetic) / (len(self.test_patients) + len(self.synthetic_patients))
        results["overall_accuracy"] = overall_accuracy
        
        # Save results
        self.results["accuracy"] = results
        
        # Generate accuracy report
        self._generate_accuracy_report(results)
        
        return results
    
    def run_performance_tests(self, batch_sizes: List[int] = [1, 5, 10, 20]) -> Dict:
        """Run performance tests on the model"""
        
        logger.info("Running performance tests...")
        
        results = {
            "single_inference": {},
            "batch_inference": {},
            "memory_usage": {},
            "cpu_usage": {}
        }
        
        # Test single inference performance
        single_times = []
        for i, patient in enumerate(self.synthetic_patients[:20]):
            logger.info(f"Testing single inference performance {i+1}/20")
            
            # Measure inference time
            start_time = time.time()
            self.model.diagnose_rare_disease(patient)
            inference_time = time.time() - start_time
            
            single_times.append(inference_time)
        
        # Calculate statistics
        results["single_inference"] = {
            "mean_time": np.mean(single_times),
            "median_time": np.median(single_times),
            "min_time": np.min(single_times),
            "max_time": np.max(single_times),
            "std_time": np.std(single_times)
        }
        
        # Test batch inference performance
        for batch_size in batch_sizes:
            logger.info(f"Testing batch inference performance with batch size {batch_size}")
            
            # Create batch
            batch = self.synthetic_patients[:batch_size]
            
            # Measure inference time
            start_time = time.time()
            self.model.batch_diagnose(batch)
            batch_time = time.time() - start_time
            
            results["batch_inference"][batch_size] = {
                "total_time": batch_time,
                "time_per_patient": batch_time / batch_size
            }
        
        # Save results
        self.results["performance"] = results
        
        # Generate performance report
        self._generate_performance_report(results)
        
        return results
    
    def run_robustness_tests(self) -> Dict:
        """Run robustness tests on the model"""
        
        logger.info("Running robustness tests...")
        
        results = {
            "missing_data": {},
            "noisy_data": {},
            "edge_cases": {}
        }
        
        # Test with missing data
        for field in ["symptoms", "clinical_notes", "age", "gender", "lab_values"]:
            logger.info(f"Testing robustness with missing {field}")
            
            # Create test patient with missing field
            patient = self.synthetic_patients[0].copy()
            
            if field == "symptoms":
                patient["symptoms"] = []
            elif field == "clinical_notes":
                patient["clinical_notes"] = ""
            elif field == "age":
                patient["age"] = 0
            elif field == "gender":
                patient["gender"] = ""
            elif field == "lab_values":
                patient["lab_values"] = {}
            
            # Get model prediction
            try:
                diagnosis = self.model.diagnose_rare_disease(patient)
                results["missing_data"][field] = {
                    "success": True,
                    "diagnosis": diagnosis["primary_diagnosis"],
                    "confidence": diagnosis["confidence"]
                }
            except Exception as e:
                results["missing_data"][field] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test with noisy data
        noise_levels = [0.1, 0.3, 0.5]
        for noise_level in noise_levels:
            logger.info(f"Testing robustness with noise level {noise_level}")
            
            # Create test patients with noisy symptoms
            correct_predictions = 0
            total_predictions = 0
            
            for i, patient in enumerate(self.synthetic_patients[:10]):
                noisy_patient = self._add_noise_to_patient(patient, noise_level)
                expected = patient["expected_diagnosis"]
                
                try:
                    diagnosis = self.model.diagnose_rare_disease(noisy_patient)
                    predicted = diagnosis["primary_diagnosis"]
                    
                    is_correct = expected in predicted
                    if is_correct:
                        correct_predictions += 1
                    
                    total_predictions += 1
                    
                except Exception as e:
                    logger.warning(f"Error with noisy patient {i}: {e}")
            
            # Calculate accuracy with noisy data
            if total_predictions > 0:
                noisy_accuracy = correct_predictions / total_predictions
            else:
                noisy_accuracy = 0
                
            results["noisy_data"][noise_level] = {
                "accuracy": noisy_accuracy,
                "correct": correct_predictions,
                "total": total_predictions
            }
        
        # Test with edge cases
        for i, patient in enumerate(self.edge_cases):
            logger.info(f"Testing edge case {i+1}/{len(self.edge_cases)}")
            
            case_type = patient["case_type"]
            
            try:
                diagnosis = self.model.diagnose_rare_disease(patient)
                results["edge_cases"][case_type] = {
                    "success": True,
                    "diagnosis": diagnosis["primary_diagnosis"],
                    "confidence": diagnosis["confidence"]
                }
            except Exception as e:
                results["edge_cases"][case_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Save results
        self.results["robustness"] = results
        
        # Generate robustness report
        self._generate_robustness_report(results)
        
        return results
    
    def run_privacy_tests(self) -> Dict:
        """Run privacy tests on the model"""
        
        logger.info("Running privacy tests...")
        
        results = {
            "membership_inference": {},
            "model_inversion": {},
            "differential_privacy": {}
        }
        
        # Simulate membership inference attack
        logger.info("Simulating membership inference attack")
        
        # Create two sets of patients: "training" and "non-training"
        training_patients = self.synthetic_patients[:10]
        non_training_patients = self.synthetic_patients[10:20]
        
        # Test if model behaves differently on "training" vs "non-training" data
        training_confidences = []
        non_training_confidences = []
        
        for patient in training_patients:
            diagnosis = self.model.diagnose_rare_disease(patient)
            training_confidences.append(diagnosis["confidence"])
        
        for patient in non_training_patients:
            diagnosis = self.model.diagnose_rare_disease(patient)
            non_training_confidences.append(diagnosis["confidence"])
        
        # Calculate statistics
        results["membership_inference"] = {
            "training_mean_confidence": np.mean(training_confidences),
            "non_training_mean_confidence": np.mean(non_training_confidences),
            "confidence_difference": np.mean(training_confidences) - np.mean(non_training_confidences),
            "vulnerability_score": abs(np.mean(training_confidences) - np.mean(non_training_confidences))
        }
        
        # Save results
        self.results["privacy"] = results
        
        # Generate privacy report
        self._generate_privacy_report(results)
        
        return results
    
    def run_explainability_tests(self) -> Dict:
        """Run explainability tests on the model"""
        
        logger.info("Running explainability tests...")
        
        results = {
            "feature_importance": {},
            "recommendation_quality": {}
        }
        
        # Test feature importance by removing features
        logger.info("Testing feature importance")
        
        base_patient = self.synthetic_patients[0]
        base_diagnosis = self.model.diagnose_rare_disease(base_patient)
        base_prediction = base_diagnosis["primary_diagnosis"]
        
        feature_impacts = {}
        
        # Test impact of removing symptoms
        for symptom in base_patient["symptoms"]:
            modified_patient = base_patient.copy()
            modified_patient["symptoms"] = [s for s in base_patient["symptoms"] if s != symptom]
            
            modified_diagnosis = self.model.diagnose_rare_disease(modified_patient)
            modified_prediction = modified_diagnosis["primary_diagnosis"]
            
            # Calculate impact
            impact = abs(modified_diagnosis["confidence"] - base_diagnosis["confidence"])
            prediction_changed = modified_prediction != base_prediction
            
            feature_impacts[f"symptom:{symptom}"] = {
                "impact": impact,
                "prediction_changed": prediction_changed
            }
        
        # Test impact of removing clinical notes
        modified_patient = base_patient.copy()
        modified_patient["clinical_notes"] = ""
        
        modified_diagnosis = self.model.diagnose_rare_disease(modified_patient)
        modified_prediction = modified_diagnosis["primary_diagnosis"]
        
        # Calculate impact
        impact = abs(modified_diagnosis["confidence"] - base_diagnosis["confidence"])
        prediction_changed = modified_prediction != base_prediction
        
        feature_impacts["clinical_notes"] = {
            "impact": impact,
            "prediction_changed": prediction_changed
        }
        
        # Test impact of removing lab values
        modified_patient = base_patient.copy()
        modified_patient["lab_values"] = {}
        
        modified_diagnosis = self.model.diagnose_rare_disease(modified_patient)
        modified_prediction = modified_diagnosis["primary_diagnosis"]
        
        # Calculate impact
        impact = abs(modified_diagnosis["confidence"] - base_diagnosis["confidence"])
        prediction_changed = modified_prediction != base_prediction
        
        feature_impacts["lab_values"] = {
            "impact": impact,
            "prediction_changed": prediction_changed
        }
        
        results["feature_importance"] = feature_impacts
        
        # Test recommendation quality
        logger.info("Testing recommendation quality")
        
        recommendation_quality = {}
        
        for i, patient in enumerate(self.synthetic_patients[:5]):
            diagnosis = self.model.diagnose_rare_disease(patient)
            recommendations = diagnosis["recommendations"]
            
            # Check if recommendations are relevant to the diagnosis
            relevance_score = self._evaluate_recommendation_relevance(
                diagnosis["primary_diagnosis"], 
                recommendations
            )
            
            recommendation_quality[i] = {
                "diagnosis": diagnosis["primary_diagnosis"],
                "recommendations": recommendations,
                "relevance_score": relevance_score
            }
        
        results["recommendation_quality"] = recommendation_quality
        
        # Save results
        self.results["explainability"] = results
        
        # Generate explainability report
        self._generate_explainability_report(results)
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run all tests"""
        
        logger.info("Running all tests...")
        
        # Run all test suites
        self.run_accuracy_tests()
        self.run_performance_tests()
        self.run_robustness_tests()
        self.run_privacy_tests()
        self.run_explainability_tests()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        return self.results
    
    def _get_expected_diagnosis(self, patient: Dict) -> str:
        """Get expected diagnosis based on patient symptoms"""
        
        # Simple mapping based on symptoms
        symptoms = patient.get("symptoms", [])
        
        if any(s in ["chorea", "involuntary movements", "cognitive decline"] for s in symptoms):
            return "Huntington Disease"
        elif any(s in ["chronic cough", "thick mucus", "recurrent lung infections"] for s in symptoms):
            return "Cystic Fibrosis"
        elif any(s in ["muscle weakness", "double vision", "drooping eyelids"] for s in symptoms):
            return "Myasthenia Gravis"
        elif any(s in ["muscle weakness", "muscle atrophy", "fasciculations"] for s in symptoms):
            return "Amyotrophic Lateral Sclerosis"
        elif any(s in ["liver problems", "neurological symptoms", "tremor"] for s in symptoms):
            return "Wilson Disease"
        else:
            return "Unknown"
    
    def _add_noise_to_patient(self, patient: Dict, noise_level: float) -> Dict:
        """Add noise to patient data"""
        
        noisy_patient = patient.copy()
        
        # Add noise to symptoms
        if noise_level > 0.3 and len(patient["symptoms"]) > 0:
            # Remove a symptom
            noisy_patient["symptoms"] = patient["symptoms"][:-1]
        
        if noise_level > 0.1:
            # Add a random symptom
            all_symptoms = [
                "fatigue", "headache", "dizziness", "nausea", "joint pain",
                "rash", "fever", "cough", "sore throat", "chest pain"
            ]
            random_symptom = np.random.choice(all_symptoms)
            noisy_patient["symptoms"] = noisy_patient["symptoms"] + [random_symptom]
        
        # Add noise to clinical notes
        if "clinical_notes" in patient and patient["clinical_notes"]:
            if noise_level > 0.3:
                # Add irrelevant information
                irrelevant_info = [
                    " Patient enjoys hiking.",
                    " Patient has a pet dog.",
                    " Patient is a vegetarian.",
                    " Patient recently traveled abroad."
                ]
                noisy_patient["clinical_notes"] = patient["clinical_notes"] + np.random.choice(irrelevant_info)
            
            if noise_level > 0.5:
                # Add contradictory information
                contradictions = [
                    " Patient denies all symptoms.",
                    " All lab tests are normal.",
                    " No significant medical history."
                ]
                noisy_patient["clinical_notes"] = patient["clinical_notes"] + np.random.choice(contradictions)
        
        # Add noise to lab values
        if "lab_values" in patient and patient["lab_values"]:
            noisy_lab_values = {}
            for test, value in patient["lab_values"].items():
                # Add random noise to lab values
                noise = np.random.uniform(-value * noise_level, value * noise_level)
                noisy_lab_values[test] = value + noise
            
            noisy_patient["lab_values"] = noisy_lab_values
        
        return noisy_patient
    
    def _evaluate_recommendation_relevance(self, diagnosis: str, recommendations: List[str]) -> float:
        """Evaluate the relevance of recommendations to the diagnosis"""
        
        # Define relevant keywords for each disease
        relevance_keywords = {
            "Huntington Disease": ["genetic", "neurological", "movement", "psychiatric", "counseling"],
            "Cystic Fibrosis": ["respiratory", "pulmonary", "pancreatic", "enzyme", "sweat", "genetic"],
            "Myasthenia Gravis": ["acetylcholine", "neuromuscular", "thymoma", "autoimmune", "EMG"],
            "Amyotrophic Lateral Sclerosis": ["ALS", "neurological", "EMG", "respiratory", "speech"],
            "Wilson Disease": ["copper", "liver", "neurological", "psychiatric", "genetic"],
            "Fabry Disease": ["enzyme", "genetic", "kidney", "cardiac", "pain"],
            "Gaucher Disease": ["enzyme", "genetic", "spleen", "liver", "bone"],
            "Pompe Disease": ["enzyme", "genetic", "muscle", "cardiac", "respiratory"],
            "Tay-Sachs Disease": ["enzyme", "genetic", "neurological", "developmental"]
        }
        
        # Get relevant keywords for the diagnosis
        keywords = relevance_keywords.get(diagnosis, [])
        if not keywords:
            return 0.5  # Default relevance score
        
        # Count how many recommendations contain relevant keywords
        relevant_count = 0
        for recommendation in recommendations:
            if any(keyword.lower() in recommendation.lower() for keyword in keywords):
                relevant_count += 1
        
        # Calculate relevance score
        if recommendations:
            return relevant_count / len(recommendations)
        else:
            return 0.0
    
    def _generate_accuracy_report(self, results: Dict) -> None:
        """Generate accuracy report"""
        
        report_path = os.path.join(self.output_dir, "accuracy_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MEDCHAIN AI ACCURACY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}%\n")
            f.write(f"Standard Cases Accuracy: {results['standard_accuracy']*100:.2f}%\n")
            f.write(f"Synthetic Cases Accuracy: {results['synthetic_accuracy']*100:.2f}%\n\n")
            
            f.write("Standard Cases Details:\n")
            f.write("-" * 80 + "\n")
            for i, case in results["standard_cases"].items():
                f.write(f"Case {i+1}:\n")
                f.write(f"  Expected: {case['expected']}\n")
                f.write(f"  Predicted: {case['predicted']}\n")
                f.write(f"  Confidence: {case['confidence']:.3f}\n")
                f.write(f"  Correct: {case['correct']}\n")
                f.write(f"  Inference Time: {case['inference_time']*1000:.2f}ms\n\n")
            
            f.write("Edge Cases Summary:\n")
            f.write("-" * 80 + "\n")
            for i, case in results["edge_cases"].items():
                f.write(f"Case Type: {case['case_type']}\n")
                f.write(f"  Expected: {case['expected']}\n")
                f.write(f"  Predicted: {case['predicted']}\n")
                f.write(f"  Confidence: {case['confidence']:.3f}\n\n")
            
            f.write("Adversarial Cases Summary:\n")
            f.write("-" * 80 + "\n")
            for i, case in results["adversarial_cases"].items():
                f.write(f"Case Type: {case['case_type']}\n")
                f.write(f"  Expected: {case['expected']}\n")
                f.write(f"  Predicted: {case['predicted']}\n")
                f.write(f"  Confidence: {case['confidence']:.3f}\n\n")
        
        logger.info(f"Accuracy report generated: {report_path}")
    
    def _generate_performance_report(self, results: Dict) -> None:
        """Generate performance report"""
        
        report_path = os.path.join(self.output_dir, "performance_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MEDCHAIN AI PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Single Inference Performance:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Time: {results['single_inference']['mean_time']*1000:.2f}ms\n")
            f.write(f"Median Time: {results['single_inference']['median_time']*1000:.2f}ms\n")
            f.write(f"Min Time: {results['single_inference']['min_time']*1000:.2f}ms\n")
            f.write(f"Max Time: {results['single_inference']['max_time']*1000:.2f}ms\n")
            f.write(f"Std Dev: {results['single_inference']['std_time']*1000:.2f}ms\n\n")
            
            f.write("Batch Inference Performance:\n")
            f.write("-" * 80 + "\n")
            for batch_size, batch_results in results["batch_inference"].items():
                f.write(f"Batch Size: {batch_size}\n")
                f.write(f"  Total Time: {batch_results['total_time']*1000:.2f}ms\n")
                f.write(f"  Time Per Patient: {batch_results['time_per_patient']*1000:.2f}ms\n\n")
        
        logger.info(f"Performance report generated: {report_path}")
    
    def _generate_robustness_report(self, results: Dict) -> None:
        """Generate robustness report"""
        
        report_path = os.path.join(self.output_dir, "robustness_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MEDCHAIN AI ROBUSTNESS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Missing Data Tests:\n")
            f.write("-" * 80 + "\n")
            for field, field_results in results["missing_data"].items():
                f.write(f"Field: {field}\n")
                f.write(f"  Success: {field_results['success']}\n")
                if field_results['success']:
                    f.write(f"  Diagnosis: {field_results['diagnosis']}\n")
                    f.write(f"  Confidence: {field_results['confidence']:.3f}\n")
                else:
                    f.write(f"  Error: {field_results['error']}\n")
                f.write("\n")
            
            f.write("Noisy Data Tests:\n")
            f.write("-" * 80 + "\n")
            for noise_level, noise_results in results["noisy_data"].items():
                f.write(f"Noise Level: {noise_level}\n")
                f.write(f"  Accuracy: {noise_results['accuracy']*100:.2f}%\n")
                f.write(f"  Correct: {noise_results['correct']}/{noise_results['total']}\n\n")
            
            f.write("Edge Cases Tests:\n")
            f.write("-" * 80 + "\n")
            for case_type, case_results in results["edge_cases"].items():
                f.write(f"Case Type: {case_type}\n")
                f.write(f"  Success: {case_results['success']}\n")
                if case_results['success']:
                    f.write(f"  Diagnosis: {case_results['diagnosis']}\n")
                    f.write(f"  Confidence: {case_results['confidence']:.3f}\n")
                else:
                    f.write(f"  Error: {case_results['error']}\n")
                f.write("\n")
        
        logger.info(f"Robustness report generated: {report_path}")
    
    def _generate_privacy_report(self, results: Dict) -> None:
        """Generate privacy report"""
        
        report_path = os.path.join(self.output_dir, "privacy_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MEDCHAIN AI PRIVACY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Membership Inference Attack Simulation:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Training Mean Confidence: {results['membership_inference']['training_mean_confidence']:.3f}\n")
            f.write(f"Non-Training Mean Confidence: {results['membership_inference']['non_training_mean_confidence']:.3f}\n")
            f.write(f"Confidence Difference: {results['membership_inference']['confidence_difference']:.3f}\n")
            f.write(f"Vulnerability Score: {results['membership_inference']['vulnerability_score']:.3f}\n\n")
            
            f.write("Privacy Assessment:\n")
            f.write("-" * 80 + "\n")
            vulnerability = results['membership_inference']['vulnerability_score']
            if vulnerability < 0.05:
                f.write("Low vulnerability to membership inference attacks.\n")
            elif vulnerability < 0.1:
                f.write("Moderate vulnerability to membership inference attacks.\n")
            else:
                f.write("High vulnerability to membership inference attacks.\n")
            f.write("\n")
        
        logger.info(f"Privacy report generated: {report_path}")
    
    def _generate_explainability_report(self, results: Dict) -> None:
        """Generate explainability report"""
        
        report_path = os.path.join(self.output_dir, "explainability_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MEDCHAIN AI EXPLAINABILITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Feature Importance:\n")
            f.write("-" * 80 + "\n")
            
            # Sort features by impact
            sorted_features = sorted(
                results["feature_importance"].items(),
                key=lambda x: x[1]["impact"],
                reverse=True
            )
            
            for feature, feature_results in sorted_features:
                f.write(f"Feature: {feature}\n")
                f.write(f"  Impact: {feature_results['impact']:.3f}\n")
                f.write(f"  Prediction Changed: {feature_results['prediction_changed']}\n\n")
            
            f.write("Recommendation Quality:\n")
            f.write("-" * 80 + "\n")
            for i, rec_results in results["recommendation_quality"].items():
                f.write(f"Case {i+1}:\n")
                f.write(f"  Diagnosis: {rec_results['diagnosis']}\n")
                f.write(f"  Recommendations:\n")
                for rec in rec_results['recommendations']:
                    f.write(f"    - {rec}\n")
                f.write(f"  Relevance Score: {rec_results['relevance_score']:.3f}\n\n")
        
        logger.info(f"Explainability report generated: {report_path}")
    
    def _generate_comprehensive_report(self) -> None:
        """Generate comprehensive report"""
        
        report_path = os.path.join(self.output_dir, "comprehensive_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MEDCHAIN AI COMPREHENSIVE EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            
            # Accuracy summary
            if "accuracy" in self.results:
                accuracy = self.results["accuracy"].get("overall_accuracy", 0) * 100
                f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
            
            # Performance summary
            if "performance" in self.results and "single_inference" in self.results["performance"]:
                mean_time = self.results["performance"]["single_inference"].get("mean_time", 0) * 1000
                f.write(f"Mean Inference Time: {mean_time:.2f}ms\n")
            
            # Robustness summary
            if "robustness" in self.results and "noisy_data" in self.results["robustness"]:
                noise_levels = sorted(self.results["robustness"]["noisy_data"].keys())
                if noise_levels:
                    max_noise = max(noise_levels)
                    max_noise_accuracy = self.results["robustness"]["noisy_data"][max_noise].get("accuracy", 0) * 100
                    f.write(f"Accuracy with {max_noise*100}% Noise: {max_noise_accuracy:.2f}%\n")
            
            # Privacy summary
            if "privacy" in self.results and "membership_inference" in self.results["privacy"]:
                vulnerability = self.results["privacy"]["membership_inference"].get("vulnerability_score", 0)
                f.write(f"Privacy Vulnerability Score: {vulnerability:.3f}\n")
            
            # Explainability summary
            if "explainability" in self.results and "recommendation_quality" in self.results["explainability"]:
                relevance_scores = [
                    rec["relevance_score"] 
                    for rec in self.results["explainability"]["recommendation_quality"].values()
                ]
                if relevance_scores:
                    mean_relevance = np.mean(relevance_scores)
                    f.write(f"Mean Recommendation Relevance: {mean_relevance:.3f}\n")
            
            f.write("\n")
            
            # Detailed sections
            f.write("DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Accuracy details
            f.write("1. ACCURACY\n")
            f.write("-" * 80 + "\n")
            if "accuracy" in self.results:
                f.write(f"Standard Cases Accuracy: {self.results['accuracy'].get('standard_accuracy', 0)*100:.2f}%\n")
                f.write(f"Synthetic Cases Accuracy: {self.results['accuracy'].get('synthetic_accuracy', 0)*100:.2f}%\n")
                f.write(f"Overall Accuracy: {self.results['accuracy'].get('overall_accuracy', 0)*100:.2f}%\n\n")
            
            # Performance details
            f.write("2. PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            if "performance" in self.results and "single_inference" in self.results["performance"]:
                single_inf = self.results["performance"]["single_inference"]
                f.write(f"Mean Inference Time: {single_inf.get('mean_time', 0)*1000:.2f}ms\n")
                f.write(f"Median Inference Time: {single_inf.get('median_time', 0)*1000:.2f}ms\n")
                f.write(f"Min Inference Time: {single_inf.get('min_time', 0)*1000:.2f}ms\n")
                f.write(f"Max Inference Time: {single_inf.get('max_time', 0)*1000:.2f}ms\n\n")
            
            # Robustness details
            f.write("3. ROBUSTNESS\n")
            f.write("-" * 80 + "\n")
            if "robustness" in self.results and "noisy_data" in self.results["robustness"]:
                for noise_level, noise_results in sorted(self.results["robustness"]["noisy_data"].items()):
                    f.write(f"Noise Level {noise_level*100}%: {noise_results.get('accuracy', 0)*100:.2f}% accuracy\n")
                f.write("\n")
            
            # Privacy details
            f.write("4. PRIVACY\n")
            f.write("-" * 80 + "\n")
            if "privacy" in self.results and "membership_inference" in self.results["privacy"]:
                mem_inf = self.results["privacy"]["membership_inference"]
                f.write(f"Training vs Non-Training Confidence Difference: {mem_inf.get('confidence_difference', 0):.3f}\n")
                f.write(f"Vulnerability Score: {mem_inf.get('vulnerability_score', 0):.3f}\n\n")
            
            # Explainability details
            f.write("5. EXPLAINABILITY\n")
            f.write("-" * 80 + "\n")
            if "explainability" in self.results and "feature_importance" in self.results["explainability"]:
                f.write("Top 5 Most Important Features:\n")
                sorted_features = sorted(
                    self.results["explainability"]["feature_importance"].items(),
                    key=lambda x: x[1]["impact"],
                    reverse=True
                )[:5]
                
                for feature, feature_results in sorted_features:
                    f.write(f"  - {feature}: Impact {feature_results['impact']:.3f}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            
            # Accuracy recommendations
            if "accuracy" in self.results:
                accuracy = self.results["accuracy"].get("overall_accuracy", 0)
                if accuracy < 0.7:
                    f.write("- Improve model accuracy through additional training data and feature engineering\n")
                elif accuracy < 0.85:
                    f.write("- Consider fine-tuning the model to improve accuracy on edge cases\n")
                else:
                    f.write("- Model accuracy is good; focus on maintaining performance\n")
            
            # Performance recommendations
            if "performance" in self.results and "single_inference" in self.results["performance"]:
                mean_time = self.results["performance"]["single_inference"].get("mean_time", 0)
                if mean_time > 1.0:
                    f.write("- Optimize model for faster inference (consider model quantization or distillation)\n")
                elif mean_time > 0.5:
                    f.write("- Performance is acceptable but could be improved for real-time applications\n")
                else:
                    f.write("- Model performance is good for real-time applications\n")
            
            # Robustness recommendations
            if "robustness" in self.results and "noisy_data" in self.results["robustness"]:
                noise_levels = sorted(self.results["robustness"]["noisy_data"].keys())
                if noise_levels:
                    max_noise = max(noise_levels)
                    max_noise_accuracy = self.results["robustness"]["noisy_data"][max_noise].get("accuracy", 0)
                    if max_noise_accuracy < 0.5:
                        f.write("- Improve model robustness to noisy inputs through data augmentation\n")
                    elif max_noise_accuracy < 0.7:
                        f.write("- Consider adding regularization to improve robustness\n")
                    else:
                        f.write("- Model shows good robustness to noisy inputs\n")
            
            # Privacy recommendations
            if "privacy" in self.results and "membership_inference" in self.results["privacy"]:
                vulnerability = self.results["privacy"]["membership_inference"].get("vulnerability_score", 0)
                if vulnerability > 0.1:
                    f.write("- Apply differential privacy techniques to reduce vulnerability to inference attacks\n")
                elif vulnerability > 0.05:
                    f.write("- Consider adding noise to model outputs to enhance privacy\n")
                else:
                    f.write("- Model shows good resistance to membership inference attacks\n")
            
            # Explainability recommendations
            if "explainability" in self.results and "recommendation_quality" in self.results["explainability"]:
                relevance_scores = [
                    rec["relevance_score"] 
                    for rec in self.results["explainability"]["recommendation_quality"].values()
                ]
                if relevance_scores:
                    mean_relevance = np.mean(relevance_scores)
                    if mean_relevance < 0.5:
                        f.write("- Improve recommendation quality by enhancing the recommendation generation logic\n")
                    elif mean_relevance < 0.7:
                        f.write("- Fine-tune recommendation system for better relevance\n")
                    else:
                        f.write("- Recommendation system shows good relevance to diagnoses\n")
        
        logger.info(f"Comprehensive report generated: {report_path}")
        
        # Save results as JSON
        json_path = os.path.join(self.output_dir, "test_results.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved as JSON: {json_path}")

if __name__ == "__main__":
    # Run the advanced test suite
    test_suite = AdvancedTestSuite()
    results = test_suite.run_all_tests()
    
    print("\n" + "="*80)
    print("🚀 ADVANCED TEST SUITE COMPLETED")
    print("="*80)
    print(f"Results saved to: {test_suite.output_dir}")
    print("="*80)