use crate::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// Privacy-preserving medical data operations
pub struct MedicalDataPrivacy {
    anonymization_map: HashMap<String, String>,
    k_anonymity_threshold: u32,
    l_diversity_threshold: u32,
}

impl MedicalDataPrivacy {
    pub fn new(k_anonymity: u32, l_diversity: u32) -> Self {
        MedicalDataPrivacy {
            anonymization_map: HashMap::new(),
            k_anonymity_threshold: k_anonymity,
            l_diversity_threshold: l_diversity,
        }
    }

    // K-anonymity implementation for medical datasets
    pub fn apply_k_anonymity(&mut self, dataset: &mut MedicalDataset) -> Result<(), String> {
        // Group patients by quasi-identifiers (age, gender, zip code)
        let mut groups = HashMap::new();
        
        for patient in &dataset.patients {
            let quasi_id = self.extract_quasi_identifiers(patient);
            groups.entry(quasi_id).or_insert_with(Vec::new).push(patient.id.clone());
        }
        
        // Generalize groups that don't meet k-anonymity threshold
        for (quasi_id, patient_ids) in groups {
            if patient_ids.len() < self.k_anonymity_threshold as usize {
                self.generalize_quasi_identifiers(&mut dataset.patients, &patient_ids)?;
            }
        }
        
        Ok(())
    }

    // L-diversity implementation
    pub fn apply_l_diversity(&mut self, dataset: &mut MedicalDataset) -> Result<(), String> {
        // Group by quasi-identifiers and check sensitive attribute diversity
        let mut groups = HashMap::new();
        
        for condition in &dataset.conditions {
            if let Some(patient_ref) = &condition.subject.reference {
                let patient_id = self.extract_patient_id_from_reference(patient_ref);
                if let Some(patient) = dataset.patients.iter().find(|p| p.id == patient_id) {
                    let quasi_id = self.extract_quasi_identifiers(patient);
                    groups.entry(quasi_id).or_insert_with(Vec::new).push(condition.clone());
                }
            }
        }
        
        // Check l-diversity for each group
        for (quasi_id, conditions) in groups {
            let unique_conditions = self.count_unique_conditions(&conditions);
            if unique_conditions < self.l_diversity_threshold {
                // Apply suppression or generalization
                self.suppress_sensitive_attributes(&mut dataset.conditions, &conditions)?;
            }
        }
        
        Ok(())
    }

    // T-closeness implementation
    pub fn apply_t_closeness(&self, dataset: &mut MedicalDataset, t_threshold: f64) -> Result<(), String> {
        // Calculate global distribution of sensitive attributes
        let global_distribution = self.calculate_global_condition_distribution(&dataset.conditions);
        
        // Group by quasi-identifiers
        let mut groups = HashMap::new();
        for condition in &dataset.conditions {
            if let Some(patient_ref) = &condition.subject.reference {
                let patient_id = self.extract_patient_id_from_reference(patient_ref);
                if let Some(patient) = dataset.patients.iter().find(|p| p.id == patient_id) {
                    let quasi_id = self.extract_quasi_identifiers(patient);
                    groups.entry(quasi_id).or_insert_with(Vec::new).push(condition.clone());
                }
            }
        }
        
        // Check t-closeness for each group
        for (quasi_id, conditions) in groups {
            let local_distribution = self.calculate_local_condition_distribution(&conditions);
            let distance = self.calculate_earth_movers_distance(&global_distribution, &local_distribution);
            
            if distance > t_threshold {
                // Apply noise injection or record suppression
                self.inject_noise_for_t_closeness(&mut dataset.conditions, &conditions)?;
            }
        }
        
        Ok(())
    }

    // Safe Harbor de-identification (HIPAA)
    pub fn apply_safe_harbor_deidentification(&mut self, dataset: &mut MedicalDataset) -> Result<(), String> {
        // Remove or generalize 18 HIPAA identifiers
        for patient in &mut dataset.patients {
            // 1. Names
            for name in &mut patient.name {
                name.family = Some("REMOVED".to_string());
                name.given = vec!["REMOVED".to_string()];
                name.text = Some("REMOVED".to_string());
            }
            
            // 2. Geographic subdivisions smaller than state
            for address in &mut patient.address {
                address.line.clear();
                address.city = Some("REMOVED".to_string());
                address.postal_code = None;
                // Keep only state
            }
            
            // 3. Dates (except year) - generalize to year only
            if let Some(ref birth_date) = patient.birth_date {
                if birth_date.len() >= 4 {
                    patient.birth_date = Some(format!("{}-01-01", &birth_date[..4]));
                }
            }
            
            // 4. Telephone numbers
            // 5. Fax numbers  
            // 6. Email addresses
            patient.contact.clear();
            
            // 7. Social security numbers
            // 8. Medical record numbers
            // 9. Health plan beneficiary numbers
            // 10. Account numbers
            // 11. Certificate/license numbers
            // 12. Vehicle identifiers and serial numbers
            // 13. Device identifiers and serial numbers
            // 14. Web URLs
            // 15. Internet protocol addresses
            // 16. Biometric identifiers
            // 17. Full-face photographs
            // 18. Any other unique identifying number
            patient.identifier.clear();
        }
        
        // Handle dates in observations and conditions
        for observation in &mut dataset.observations {
            if let Some(ref date) = observation.effective_datetime {
                if date.len() >= 4 {
                    observation.effective_datetime = Some(format!("{}-01-01", &date[..4]));
                }
            }
        }
        
        for condition in &mut dataset.conditions {
            if let Some(ref date) = condition.recorded_date {
                if date.len() >= 4 {
                    condition.recorded_date = Some(format!("{}-01-01", &date[..4]));
                }
            }
        }
        
        Ok(())
    }

    // Differential privacy for medical data
    pub fn apply_differential_privacy(&self, dataset: &mut MedicalDataset, epsilon: f64) -> Result<(), String> {
        // Add Laplace noise to numerical observations
        for observation in &mut dataset.observations {
            if let Some(ref mut value) = observation.value {
                match value {
                    ObservationValue::Quantity(ref mut quantity) => {
                        if let Some(ref mut val) = quantity.value {
                            let sensitivity = self.estimate_sensitivity(&observation.code);
                            let noise = self.sample_laplace_noise(0.0, sensitivity / epsilon);
                            *val += noise;
                        }
                    }
                    ObservationValue::Integer(ref mut int_val) => {
                        let sensitivity = 1.0; // For count data
                        let noise = self.sample_laplace_noise(0.0, sensitivity / epsilon);
                        *int_val = (*int_val as f64 + noise).round() as i32;
                    }
                    _ => {} // No noise for non-numerical values
                }
            }
        }
        
        Ok(())
    }

    // Synthetic data generation for privacy
    pub fn generate_synthetic_dataset(&self, original: &MedicalDataset, num_synthetic: usize) -> Result<MedicalDataset, String> {
        let mut synthetic_dataset = MedicalDataset::new(
            format!("{}_synthetic", original.id),
            format!("{} (Synthetic)", original.name),
            "Synthetic dataset generated for privacy preservation".to_string(),
        );
        
        // Generate synthetic patients based on statistical properties
        for i in 0..num_synthetic {
            let synthetic_patient = self.generate_synthetic_patient(&original.patients, i)?;
            synthetic_dataset.add_patient(synthetic_patient)?;
        }
        
        // Generate synthetic observations
        let observations_per_patient = if !original.patients.is_empty() {
            original.observations.len() / original.patients.len()
        } else {
            0
        };
        
        for patient in &synthetic_dataset.patients {
            for _ in 0..observations_per_patient {
                let synthetic_observation = self.generate_synthetic_observation(&original.observations, &patient.id)?;
                synthetic_dataset.add_observation(synthetic_observation)?;
            }
        }
        
        Ok(synthetic_dataset)
    }

    // Helper methods
    fn extract_quasi_identifiers(&self, patient: &Patient) -> String {
        let age = self.calculate_age_from_birth_date(&patient.birth_date);
        let gender = match &patient.gender {
            Some(Gender::Male) => "M",
            Some(Gender::Female) => "F",
            _ => "U",
        };
        let zip = patient.address.first()
            .and_then(|addr| addr.postal_code.as_ref())
            .map(|zip| &zip[..3.min(zip.len())]) // First 3 digits of zip
            .unwrap_or("000");
        
        format!("{}_{}_{}_{}", age / 10 * 10, gender, zip, "")
    }

    fn calculate_age_from_birth_date(&self, birth_date: &Option<String>) -> u32 {
        if let Some(date_str) = birth_date {
            if let Ok(birth) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                let today = chrono::Utc::now().date_naive();
                return today.years_since(birth).unwrap_or(0);
            }
        }
        0
    }

    fn generalize_quasi_identifiers(&self, patients: &mut [Patient], patient_ids: &[String]) -> Result<(), String> {
        for patient in patients.iter_mut() {
            if patient_ids.contains(&patient.id) {
                // Generalize age to age ranges
                if let Some(ref birth_date) = patient.birth_date {
                    let age = self.calculate_age_from_birth_date(&Some(birth_date.clone()));
                    let age_range = (age / 10) * 10;
                    // Set birth date to beginning of age range decade
                    let current_year = chrono::Utc::now().year() as u32;
                    let birth_year = current_year - age_range;
                    patient.birth_date = Some(format!("{}-01-01", birth_year));
                }
                
                // Generalize postal codes
                for address in &mut patient.address {
                    if let Some(ref postal_code) = address.postal_code {
                        if postal_code.len() >= 3 {
                            address.postal_code = Some(format!("{}00", &postal_code[..3]));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn extract_patient_id_from_reference(&self, reference: &str) -> String {
        reference.split('/').last().unwrap_or(reference).to_string()
    }

    fn count_unique_conditions(&self, conditions: &[Condition]) -> u32 {
        let mut unique_codes = std::collections::HashSet::new();
        for condition in conditions {
            if let Some(ref code) = condition.code {
                if let Some(ref text) = code.text {
                    unique_codes.insert(text.clone());
                }
            }
        }
        unique_codes.len() as u32
    }

    fn suppress_sensitive_attributes(&self, all_conditions: &mut [Condition], target_conditions: &[Condition]) -> Result<(), String> {
        let target_ids: std::collections::HashSet<String> = target_conditions.iter().map(|c| c.id.clone()).collect();
        
        for condition in all_conditions.iter_mut() {
            if target_ids.contains(&condition.id) {
                // Suppress by generalizing the condition
                if let Some(ref mut code) = condition.code {
                    code.text = Some("GENERALIZED_CONDITION".to_string());
                    code.coding.clear();
                }
            }
        }
        Ok(())
    }

    fn calculate_global_condition_distribution(&self, conditions: &[Condition]) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();
        let total = conditions.len() as f64;
        
        for condition in conditions {
            if let Some(ref code) = condition.code {
                if let Some(ref text) = code.text {
                    *distribution.entry(text.clone()).or_insert(0.0) += 1.0 / total;
                }
            }
        }
        
        distribution
    }

    fn calculate_local_condition_distribution(&self, conditions: &[Condition]) -> HashMap<String, f64> {
        self.calculate_global_condition_distribution(conditions)
    }

    fn calculate_earth_movers_distance(&self, dist1: &HashMap<String, f64>, dist2: &HashMap<String, f64>) -> f64 {
        // Simplified Earth Mover's Distance calculation
        let mut distance = 0.0;
        let all_keys: std::collections::HashSet<String> = dist1.keys().chain(dist2.keys()).cloned().collect();
        
        for key in all_keys {
            let p1 = dist1.get(&key).unwrap_or(&0.0);
            let p2 = dist2.get(&key).unwrap_or(&0.0);
            distance += (p1 - p2).abs();
        }
        
        distance / 2.0
    }

    fn inject_noise_for_t_closeness(&self, all_conditions: &mut [Condition], _target_conditions: &[Condition]) -> Result<(), String> {
        // Simplified noise injection - in practice would use more sophisticated methods
        for condition in all_conditions.iter_mut() {
            if let Some(ref mut code) = condition.code {
                // Add some randomness to condition codes
                if let Some(ref mut text) = code.text {
                    if rand::random::<f64>() < 0.1 { // 10% chance to modify
                        *text = format!("{}_MODIFIED", text);
                    }
                }
            }
        }
        Ok(())
    }

    fn estimate_sensitivity(&self, observation_code: &CodeableConcept) -> f64 {
        // Estimate sensitivity based on observation type
        if let Some(ref text) = observation_code.text {
            match text.to_lowercase().as_str() {
                s if s.contains("glucose") => 50.0,
                s if s.contains("blood pressure") => 20.0,
                s if s.contains("heart rate") => 30.0,
                s if s.contains("temperature") => 2.0,
                _ => 10.0, // Default sensitivity
            }
        } else {
            10.0
        }
    }

    fn sample_laplace_noise(&self, mean: f64, scale: f64) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let u: f64 = rng.gen_range(-0.5..0.5);
        mean - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    fn generate_synthetic_patient(&self, original_patients: &[Patient], index: usize) -> Result<Patient, String> {
        let mut synthetic_patient = Patient::new(format!("synthetic_patient_{}", index));
        
        // Generate synthetic demographics based on original distribution
        let age_distribution = self.calculate_age_distribution(original_patients);
        let gender_distribution = self.calculate_gender_distribution(original_patients);
        
        // Sample from distributions
        let synthetic_age = self.sample_from_age_distribution(&age_distribution);
        let synthetic_gender = self.sample_from_gender_distribution(&gender_distribution);
        
        synthetic_patient.set_gender(synthetic_gender);
        
        // Set synthetic birth date based on age
        let current_year = chrono::Utc::now().year() as u32;
        let birth_year = current_year - synthetic_age;
        synthetic_patient.set_birth_date(format!("{}-01-01", birth_year));
        
        // Add synthetic name
        let synthetic_name = HumanName {
            use_type: Some("official".to_string()),
            text: Some(format!("Synthetic Patient {}", index)),
            family: Some(format!("Patient{}", index)),
            given: vec!["Synthetic".to_string()],
            prefix: Vec::new(),
            suffix: Vec::new(),
            period: None,
        };
        synthetic_patient.add_name(synthetic_name);
        
        Ok(synthetic_patient)
    }

    fn generate_synthetic_observation(&self, original_observations: &[Observation], patient_id: &str) -> Result<Observation, String> {
        if original_observations.is_empty() {
            return Err("No original observations to base synthesis on".to_string());
        }
        
        // Sample a random observation type from originals
        let template = &original_observations[rand::random::<usize>() % original_observations.len()];
        
        let mut synthetic_obs = Observation::new(
            format!("synthetic_obs_{}_{}", patient_id, uuid::Uuid::new_v4()),
            template.code.clone(),
            create_reference(&format!("Patient/{}", patient_id), None),
        );
        
        // Generate synthetic value based on original distribution
        if let Some(ref original_value) = template.value {
            let synthetic_value = match original_value {
                ObservationValue::Quantity(ref q) => {
                    if let Some(val) = q.value {
                        let noise = self.sample_laplace_noise(0.0, val * 0.1); // 10% noise
                        ObservationValue::Quantity(create_quantity(val + noise, 
                            q.unit.as_ref().unwrap_or(&"".to_string()), None, None))
                    } else {
                        original_value.clone()
                    }
                }
                _ => original_value.clone(),
            };
            synthetic_obs.set_value(synthetic_value);
        }
        
        Ok(synthetic_obs)
    }

    fn calculate_age_distribution(&self, patients: &[Patient]) -> Vec<u32> {
        patients.iter()
            .map(|p| self.calculate_age_from_birth_date(&p.birth_date))
            .collect()
    }

    fn calculate_gender_distribution(&self, patients: &[Patient]) -> HashMap<Gender, f64> {
        let mut distribution = HashMap::new();
        let total = patients.len() as f64;
        
        for patient in patients {
            let gender = patient.gender.clone().unwrap_or(Gender::Unknown);
            *distribution.entry(gender).or_insert(0.0) += 1.0 / total;
        }
        
        distribution
    }

    fn sample_from_age_distribution(&self, ages: &[u32]) -> u32 {
        if ages.is_empty() {
            return 30; // Default age
        }
        ages[rand::random::<usize>() % ages.len()]
    }

    fn sample_from_gender_distribution(&self, distribution: &HashMap<Gender, f64>) -> Gender {
        let random_val: f64 = rand::random();
        let mut cumulative = 0.0;
        
        for (gender, probability) in distribution {
            cumulative += probability;
            if random_val <= cumulative {
                return gender.clone();
            }
        }
        
        Gender::Unknown
    }
}

// Privacy metrics and reporting
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct PrivacyMetrics {
    pub k_anonymity_level: u32,
    pub l_diversity_level: u32,
    pub t_closeness_threshold: f64,
    pub differential_privacy_epsilon: f64,
    pub information_loss: f64,
    pub utility_preservation: f64,
    pub re_identification_risk: f64,
}

impl PrivacyMetrics {
    pub fn calculate_for_dataset(dataset: &MedicalDataset) -> Self {
        PrivacyMetrics {
            k_anonymity_level: Self::calculate_k_anonymity(dataset),
            l_diversity_level: Self::calculate_l_diversity(dataset),
            t_closeness_threshold: Self::calculate_t_closeness(dataset),
            differential_privacy_epsilon: 0.0, // Would be set based on applied DP
            information_loss: Self::calculate_information_loss(dataset),
            utility_preservation: Self::calculate_utility_preservation(dataset),
            re_identification_risk: Self::calculate_reidentification_risk(dataset),
        }
    }

    fn calculate_k_anonymity(dataset: &MedicalDataset) -> u32 {
        // Simplified k-anonymity calculation
        let mut min_group_size = u32::MAX;
        let mut groups = HashMap::new();
        
        for patient in &dataset.patients {
            let quasi_id = format!("{}_{:?}", 
                patient.birth_date.as_ref().unwrap_or(&"unknown".to_string())[..4].to_string(),
                patient.gender
            );
            *groups.entry(quasi_id).or_insert(0u32) += 1;
        }
        
        for count in groups.values() {
            if *count < min_group_size {
                min_group_size = *count;
            }
        }
        
        if min_group_size == u32::MAX { 0 } else { min_group_size }
    }

    fn calculate_l_diversity(_dataset: &MedicalDataset) -> u32 {
        // Simplified l-diversity calculation
        // In practice, would analyze sensitive attribute diversity within equivalence classes
        1
    }

    fn calculate_t_closeness(_dataset: &MedicalDataset) -> f64 {
        // Simplified t-closeness calculation
        0.5
    }

    fn calculate_information_loss(_dataset: &MedicalDataset) -> f64 {
        // Measure of how much information was lost during anonymization
        0.2 // 20% information loss (example)
    }

    fn calculate_utility_preservation(_dataset: &MedicalDataset) -> f64 {
        // Measure of how much utility is preserved after anonymization
        0.8 // 80% utility preserved (example)
    }

    fn calculate_reidentification_risk(_dataset: &MedicalDataset) -> f64 {
        // Estimate of re-identification risk
        0.05 // 5% risk (example)
    }
}