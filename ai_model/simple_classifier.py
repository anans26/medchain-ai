#!/usr/bin/env python3
"""
Simplified Real AI Model for Rare Disease Classification
Lightweight version that can be converted to ONNX for Rust integration
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRareDiseaseClassifier(nn.Module):
    """
    Simplified but real neural network for rare disease classification
    Uses symptom embeddings + clinical features
    """
    
    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 128, num_diseases: int = 50):
        super().__init__()
        
        # Symptom embedding layer
        self.symptom_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Symptom encoder (processes sequence of symptoms)
        self.symptom_encoder = nn.LSTM(
            embedding_dim, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        
        # Lab values encoder
        self.lab_encoder = nn.Sequential(
            nn.Linear(20, 64),  # 20 key lab tests
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Demographics encoder
        self.demo_encoder = nn.Sequential(
            nn.Linear(5, 16),  # Age, gender, BMI, etc.
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32 + 16, 128),  # Concatenated features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_diseases)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(64 + 32 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.num_diseases = num_diseases
        
    def forward(self, symptom_ids, lab_values, demographics):
        """
        Forward pass
        
        Args:
            symptom_ids: Symptom token IDs [batch_size, seq_len]
            lab_values: Lab test results [batch_size, 20]
            demographics: Patient demographics [batch_size, 5]
        """
        
        # Encode symptoms
        symptom_embeds = self.symptom_embedding(symptom_ids)
        lstm_out, (hidden, _) = self.symptom_encoder(symptom_embeds)
        symptom_features = hidden[-1]  # Use last hidden state
        
        # Encode lab values
        lab_features = self.lab_encoder(lab_values)
        
        # Encode demographics
        demo_features = self.demo_encoder(demographics)
        
        # Concatenate all features
        combined_features = torch.cat([symptom_features, lab_features, demo_features], dim=1)
        
        # Classification
        disease_logits = self.classifier(combined_features)
        confidence = self.confidence_head(combined_features)
        
        return disease_logits, confidence

class MedicalVocabulary:
    """Medical vocabulary for symptom encoding"""
    
    def __init__(self):
        # Common medical symptoms and terms
        self.symptoms = [
            "fever", "fatigue", "weakness", "pain", "headache", "nausea", "vomiting",
            "diarrhea", "constipation", "cough", "shortness_of_breath", "chest_pain",
            "muscle_weakness", "joint_pain", "rash", "swelling", "dizziness",
            "confusion", "memory_loss", "seizures", "tremor", "paralysis",
            "double_vision", "blurred_vision", "hearing_loss", "difficulty_swallowing",
            "speech_problems", "balance_problems", "coordination_problems",
            "involuntary_movements", "muscle_stiffness", "muscle_cramps",
            "weight_loss", "weight_gain", "loss_of_appetite", "excessive_thirst",
            "frequent_urination", "blood_in_urine", "blood_in_stool",
            "abdominal_pain", "back_pain", "neck_pain", "leg_pain", "arm_pain",
            "numbness", "tingling", "burning_sensation", "itching", "bruising",
            "bleeding", "pale_skin", "yellow_skin", "blue_lips", "cold_hands",
            # Rare disease specific symptoms
            "chorea", "dystonia", "ataxia", "spasticity", "myoclonus",
            "fasciculations", "muscle_atrophy", "contractures", "scoliosis",
            "cardiomyopathy", "arrhythmia", "heart_murmur", "enlarged_heart",
            "liver_enlargement", "spleen_enlargement", "kidney_problems",
            "cataracts", "retinal_degeneration", "optic_atrophy",
            "developmental_delay", "intellectual_disability", "autism",
            "behavioral_changes", "personality_changes", "depression", "anxiety",
            "psychosis", "hallucinations", "delusions", "sleep_problems",
            "chronic_fatigue", "exercise_intolerance", "heat_intolerance",
            "cold_intolerance", "sun_sensitivity", "drug_sensitivity",
            "recurrent_infections", "immune_deficiency", "autoimmune_symptoms",
            "inflammatory_symptoms", "allergic_reactions", "anaphylaxis",
            "growth_problems", "puberty_problems", "fertility_problems",
            "pregnancy_complications", "birth_defects", "family_history"
        ]
        
        # Create vocabulary mappings
        self.word_to_id = {word: i + 1 for i, word in enumerate(self.symptoms)}  # +1 for padding
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<UNK>"] = len(self.symptoms) + 1
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        
    def encode_symptoms(self, symptom_list: List[str], max_length: int = 20) -> List[int]:
        """Convert symptom list to token IDs"""
        
        # Normalize and tokenize
        normalized_symptoms = []
        for symptom in symptom_list:
            # Simple normalization
            normalized = symptom.lower().replace(" ", "_").replace("-", "_")
            normalized_symptoms.append(normalized)
        
        # Convert to IDs
        ids = []
        for symptom in normalized_symptoms:
            if symptom in self.word_to_id:
                ids.append(self.word_to_id[symptom])
            else:
                ids.append(self.word_to_id["<UNK>"])
        
        # Pad or truncate
        if len(ids) < max_length:
            ids.extend([0] * (max_length - len(ids)))  # Pad with 0
        else:
            ids = ids[:max_length]  # Truncate
        
        return ids

class RealMedicalDataset:
    """Generate realistic medical data for training"""
    
    def __init__(self):
        self.diseases = [
            "Huntington Disease", "Cystic Fibrosis", "Myasthenia Gravis",
            "Amyotrophic Lateral Sclerosis", "Duchenne Muscular Dystrophy",
            "Wilson Disease", "Fabry Disease", "Gaucher Disease",
            "Pompe Disease", "Tay-Sachs Disease", "Niemann-Pick Disease",
            "Marfan Syndrome", "Ehlers-Danlos Syndrome", "Osteogenesis Imperfecta",
            "Neurofibromatosis Type 1", "Tuberous Sclerosis", "Von Hippel-Lindau",
            "Hereditary Hemorrhagic Telangiectasia", "Polycystic Kidney Disease",
            "Sickle Cell Disease", "Thalassemia", "Hemophilia A", "Hemophilia B",
            "Von Willebrand Disease", "Thrombotic Thrombocytopenic Purpura",
            "Systemic Lupus Erythematosus", "Sjögren Syndrome", "Scleroderma",
            "Dermatomyositis", "Polymyositis", "Inclusion Body Myositis",
            "Myotonic Dystrophy", "Spinal Muscular Atrophy", "Charcot-Marie-Tooth",
            "Friedreich Ataxia", "Spinocerebellar Ataxia", "Multiple System Atrophy",
            "Progressive Supranuclear Palsy", "Corticobasal Degeneration",
            "Primary Progressive Aphasia", "Frontotemporal Dementia",
            "Creutzfeldt-Jakob Disease", "Fatal Familial Insomnia",
            "Gerstmann-Sträussler-Scheinker", "Kuru", "Variant Creutzfeldt-Jakob",
            "Alexander Disease", "Canavan Disease", "Krabbe Disease",
            "Metachromatic Leukodystrophy", "Adrenoleukodystrophy"
        ]
        
        # Disease-specific symptom patterns
        self.disease_symptoms = {
            "Huntington Disease": ["chorea", "involuntary_movements", "cognitive_decline", "behavioral_changes", "depression", "difficulty_swallowing"],
            "Cystic Fibrosis": ["chronic_cough", "recurrent_infections", "weight_loss", "digestive_problems", "salty_skin", "shortness_of_breath"],
            "Myasthenia Gravis": ["muscle_weakness", "double_vision", "difficulty_swallowing", "fatigue", "speech_problems", "breathing_problems"],
            "Amyotrophic Lateral Sclerosis": ["muscle_weakness", "muscle_atrophy", "fasciculations", "speech_problems", "difficulty_swallowing", "breathing_problems"],
            "Duchenne Muscular Dystrophy": ["muscle_weakness", "muscle_atrophy", "contractures", "scoliosis", "cardiomyopathy", "developmental_delay"],
            "Wilson Disease": ["liver_enlargement", "neurological_symptoms", "psychiatric_symptoms", "tremor", "dystonia", "yellow_skin"],
            "Fabry Disease": ["pain", "burning_sensation", "rash", "kidney_problems", "heart_problems", "hearing_loss"],
            "Gaucher Disease": ["fatigue", "bone_pain", "liver_enlargement", "spleen_enlargement", "bleeding", "bruising"],
            "Pompe Disease": ["muscle_weakness", "breathing_problems", "heart_problems", "feeding_problems", "developmental_delay"],
            "Tay-Sachs Disease": ["developmental_delay", "seizures", "vision_loss", "hearing_loss", "muscle_weakness", "paralysis"],
        }
        
        # Lab test normal ranges
        self.lab_ranges = {
            "hemoglobin": (12.0, 16.0, "g/dL"),
            "white_blood_cells": (4.0, 11.0, "K/uL"),
            "platelets": (150, 450, "K/uL"),
            "glucose": (70, 100, "mg/dL"),
            "creatinine": (0.6, 1.2, "mg/dL"),
            "alt": (7, 56, "U/L"),
            "ast": (10, 40, "U/L"),
            "bilirubin": (0.2, 1.2, "mg/dL"),
            "cholesterol": (125, 200, "mg/dL"),
            "triglycerides": (50, 150, "mg/dL"),
            "ldh": (140, 280, "U/L"),
            "cpk": (30, 200, "U/L"),
            "troponin": (0.0, 0.04, "ng/mL"),
            "bnp": (0, 100, "pg/mL"),
            "tsh": (0.4, 4.0, "mIU/L"),
            "t4": (4.5, 12.0, "ug/dL"),
            "vitamin_b12": (200, 900, "pg/mL"),
            "folate": (2.7, 17.0, "ng/mL"),
            "iron": (60, 170, "ug/dL"),
            "ferritin": (12, 300, "ng/mL"),
        }
        
        self.vocab = MedicalVocabulary()
        
    def generate_patient(self, disease_idx: int) -> Dict:
        """Generate realistic patient data"""
        
        disease = self.diseases[disease_idx]
        
        # Get disease-specific symptoms
        base_symptoms = self.disease_symptoms.get(disease, ["fatigue", "weakness", "pain"])
        
        # Add some random symptoms for variability
        all_symptoms = list(self.vocab.symptoms)
        additional_symptoms = np.random.choice(all_symptoms, size=np.random.randint(2, 5), replace=False)
        
        patient_symptoms = base_symptoms + list(additional_symptoms)
        
        # Generate demographics
        age = max(18, min(90, np.random.normal(50, 20)))
        gender = np.random.choice([0, 1])  # 0=female, 1=male
        bmi = max(15, min(40, np.random.normal(25, 5)))
        
        # Generate lab values
        lab_values = []
        for lab_name, (low, high, unit) in self.lab_ranges.items():
            if disease == "Wilson Disease" and lab_name in ["alt", "ast"]:
                # Liver enzymes elevated in Wilson's
                value = np.random.normal(high * 2, high * 0.3)
            elif disease == "Gaucher Disease" and lab_name == "platelets":
                # Low platelets in Gaucher
                value = np.random.normal(low * 0.5, low * 0.2)
            elif disease == "Cystic Fibrosis" and lab_name == "glucose":
                # Diabetes common in CF
                value = np.random.normal(150, 30)
            else:
                # Normal range with some variation
                value = np.random.uniform(low * 0.8, high * 1.2)
            
            lab_values.append(max(0, value))  # Ensure positive values
        
        # Normalize lab values to 0-1 range for neural network
        normalized_labs = []
        for i, (lab_name, (low, high, unit)) in enumerate(self.lab_ranges.items()):
            raw_value = lab_values[i]
            normalized = (raw_value - low) / (high - low)
            normalized_labs.append(max(0, min(2, normalized)))  # Clamp to reasonable range
        
        return {
            "disease": disease,
            "disease_idx": disease_idx,
            "symptoms": patient_symptoms,
            "symptom_ids": self.vocab.encode_symptoms(patient_symptoms),
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "demographics": [
                (age - 50) / 20,  # Normalize age
                gender,
                (bmi - 25) / 5,   # Normalize BMI
                np.random.normal(0, 1),  # Additional demographic feature
                np.random.normal(0, 1),  # Additional demographic feature
            ],
            "lab_values": normalized_labs,
            "raw_lab_values": lab_values,
        }
    
    def generate_dataset(self, samples_per_disease: int = 200) -> List[Dict]:
        """Generate complete dataset"""
        
        dataset = []
        for disease_idx in range(len(self.diseases)):
            for _ in range(samples_per_disease):
                patient = self.generate_patient(disease_idx)
                dataset.append(patient)
        
        return dataset

def train_simple_model():
    """Train the simplified model"""
    
    logger.info("Training simplified rare disease classifier...")
    
    # Generate dataset
    dataset_gen = RealMedicalDataset()
    dataset = dataset_gen.generate_dataset(samples_per_disease=100)
    
    # Split dataset
    np.random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    logger.info(f"Dataset: {len(train_data)} train, {len(val_data)} val")
    
    # Create model
    model = SimpleRareDiseaseClassifier(
        vocab_size=dataset_gen.vocab.vocab_size,
        num_diseases=len(dataset_gen.diseases)
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle training data
        np.random.shuffle(train_data)
        
        for i in range(0, len(train_data), 32):  # Batch size 32
            batch = train_data[i:i+32]
            
            # Prepare batch tensors
            symptom_ids = torch.LongTensor([p["symptom_ids"] for p in batch])
            lab_values = torch.FloatTensor([p["lab_values"] for p in batch])
            demographics = torch.FloatTensor([p["demographics"] for p in batch])
            labels = torch.LongTensor([p["disease_idx"] for p in batch])
            
            # Forward pass
            optimizer.zero_grad()
            disease_logits, confidence = model(symptom_ids, lab_values, demographics)
            loss = criterion(disease_logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(disease_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data), 32):
                batch = val_data[i:i+32]
                
                symptom_ids = torch.LongTensor([p["symptom_ids"] for p in batch])
                lab_values = torch.FloatTensor([p["lab_values"] for p in batch])
                demographics = torch.FloatTensor([p["demographics"] for p in batch])
                labels = torch.LongTensor([p["disease_idx"] for p in batch])
                
                disease_logits, confidence = model(symptom_ids, lab_values, demographics)
                _, predicted = torch.max(disease_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        model.train()
        
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/20: Loss={total_loss/len(train_data)*32:.4f}, "
                   f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset_gen.vocab.vocab_size,
        'num_diseases': len(dataset_gen.diseases),
        'diseases': dataset_gen.diseases,
        'vocab': dataset_gen.vocab.word_to_id,
    }, 'simple_rare_disease_model.pth')
    
    # Save vocabulary and disease list
    with open('medical_vocab.json', 'w') as f:
        json.dump(dataset_gen.vocab.word_to_id, f)
    
    with open('disease_list.json', 'w') as f:
        json.dump(dataset_gen.diseases, f)
    
    logger.info(f"Model saved! Final validation accuracy: {val_acc:.2f}%")
    
    return model, dataset_gen

def export_to_onnx(model, vocab_size):
    """Export model to ONNX for Rust integration"""
    
    logger.info("Exporting model to ONNX...")
    
    model.eval()
    
    # Create dummy inputs
    dummy_symptom_ids = torch.randint(0, vocab_size, (1, 20))
    dummy_lab_values = torch.randn(1, 20)
    dummy_demographics = torch.randn(1, 5)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_symptom_ids, dummy_lab_values, dummy_demographics),
        "rare_disease_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['symptom_ids', 'lab_values', 'demographics'],
        output_names=['disease_logits', 'confidence'],
        dynamic_axes={
            'symptom_ids': {0: 'batch_size'},
            'lab_values': {0: 'batch_size'},
            'demographics': {0: 'batch_size'},
            'disease_logits': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
    
    logger.info("ONNX model exported successfully!")

if __name__ == "__main__":
    # Train model
    model, dataset_gen = train_simple_model()
    
    # Export to ONNX
    export_to_onnx(model, dataset_gen.vocab.vocab_size)
    
    # Test inference
    logger.info("Testing inference...")
    
    test_patient = dataset_gen.generate_patient(0)  # Huntington's Disease
    
    model.eval()
    with torch.no_grad():
        symptom_ids = torch.LongTensor([test_patient["symptom_ids"]])
        lab_values = torch.FloatTensor([test_patient["lab_values"]])
        demographics = torch.FloatTensor([test_patient["demographics"]])
        
        disease_logits, confidence = model(symptom_ids, lab_values, demographics)
        
        predicted_idx = torch.argmax(disease_logits, dim=1).item()
        predicted_disease = dataset_gen.diseases[predicted_idx]
        confidence_score = confidence.item()
        
        logger.info(f"Test Patient:")
        logger.info(f"  True Disease: {test_patient['disease']}")
        logger.info(f"  Symptoms: {test_patient['symptoms'][:5]}")
        logger.info(f"  Predicted: {predicted_disease}")
        logger.info(f"  Confidence: {confidence_score:.4f}")
        
        # Check if prediction is correct
        if predicted_disease == test_patient['disease']:
            logger.info("✅ Correct prediction!")
        else:
            logger.info("❌ Incorrect prediction")
    
    logger.info("Model training and export completed successfully!")