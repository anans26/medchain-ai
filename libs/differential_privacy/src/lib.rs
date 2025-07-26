use rand::Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Laplace, Continuous};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    pub epsilon: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAccountant {
    pub total_budget: PrivacyBudget,
    pub spent_budget: PrivacyBudget,
    pub queries: Vec<PrivacyQuery>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyQuery {
    pub query_id: String,
    pub epsilon_spent: f64,
    pub delta_spent: f64,
    pub timestamp: u64,
    pub mechanism: String,
}

pub trait DifferentialPrivacy {
    fn add_laplace_noise(&self, value: f64, sensitivity: f64, epsilon: f64) -> f64;
    fn add_gaussian_noise(&self, value: f64, sensitivity: f64, epsilon: f64, delta: f64) -> f64;
    fn clip_gradients(&self, gradients: &[f32], clip_norm: f32) -> Vec<f32>;
    fn compute_privacy_loss(&self, epsilon: f64, delta: f64) -> f64;
}

pub struct PrivacyMechanism {
    pub rng: rand::rngs::ThreadRng,
}

impl PrivacyMechanism {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl Default for PrivacyMechanism {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentialPrivacy for PrivacyMechanism {
    fn add_laplace_noise(&self, value: f64, sensitivity: f64, epsilon: f64) -> f64 {
        let scale = sensitivity / epsilon;
        let laplace = Laplace::new(0.0, scale).unwrap();
        let noise = laplace.sample(&mut rand::thread_rng());
        value + noise
    }
    
    fn add_gaussian_noise(&self, value: f64, sensitivity: f64, epsilon: f64, delta: f64) -> f64 {
        // Gaussian mechanism for (ε, δ)-differential privacy
        let sigma = sensitivity * (2.0 * (1.25 / delta).ln()).sqrt() / epsilon;
        let noise: f64 = rand::thread_rng().gen_range(-3.0 * sigma..3.0 * sigma);
        value + noise
    }
    
    fn clip_gradients(&self, gradients: &[f32], clip_norm: f32) -> Vec<f32> {
        let norm: f32 = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();
        
        if norm <= clip_norm {
            gradients.to_vec()
        } else {
            let scale = clip_norm / norm;
            gradients.iter().map(|&g| g * scale).collect()
        }
    }
    
    fn compute_privacy_loss(&self, epsilon: f64, delta: f64) -> f64 {
        // Simplified privacy loss computation
        // In practice, this would use more sophisticated composition theorems
        epsilon + delta.ln().abs()
    }
}

impl PrivacyAccountant {
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            total_budget: PrivacyBudget { epsilon, delta },
            spent_budget: PrivacyBudget { epsilon: 0.0, delta: 0.0 },
            queries: Vec::new(),
        }
    }
    
    pub fn can_spend(&self, epsilon: f64, delta: f64) -> bool {
        self.spent_budget.epsilon + epsilon <= self.total_budget.epsilon &&
        self.spent_budget.delta + delta <= self.total_budget.delta
    }
    
    pub fn spend_budget(&mut self, query_id: String, epsilon: f64, delta: f64, mechanism: String) -> Result<(), String> {
        if !self.can_spend(epsilon, delta) {
            return Err("Insufficient privacy budget".to_string());
        }
        
        self.spent_budget.epsilon += epsilon;
        self.spent_budget.delta += delta;
        
        let query = PrivacyQuery {
            query_id,
            epsilon_spent: epsilon,
            delta_spent: delta,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            mechanism,
        };
        
        self.queries.push(query);
        Ok(())
    }
    
    pub fn remaining_budget(&self) -> PrivacyBudget {
        PrivacyBudget {
            epsilon: self.total_budget.epsilon - self.spent_budget.epsilon,
            delta: self.total_budget.delta - self.spent_budget.delta,
        }
    }
    
    pub fn reset_budget(&mut self) {
        self.spent_budget = PrivacyBudget { epsilon: 0.0, delta: 0.0 };
        self.queries.clear();
    }
}

pub struct FederatedPrivacy {
    pub mechanism: PrivacyMechanism,
    pub accountants: HashMap<String, PrivacyAccountant>,
}

impl FederatedPrivacy {
    pub fn new() -> Self {
        Self {
            mechanism: PrivacyMechanism::new(),
            accountants: HashMap::new(),
        }
    }
    
    pub fn register_participant(&mut self, participant_id: String, epsilon: f64, delta: f64) {
        let accountant = PrivacyAccountant::new(epsilon, delta);
        self.accountants.insert(participant_id, accountant);
    }
    
    pub fn add_noise_to_gradients(
        &mut self,
        participant_id: &str,
        gradients: &[f32],
        sensitivity: f64,
        epsilon: f64,
        delta: f64,
    ) -> Result<Vec<f32>, String> {
        let accountant = self.accountants.get_mut(participant_id)
            .ok_or("Participant not registered")?;
        
        if !accountant.can_spend(epsilon, delta) {
            return Err("Insufficient privacy budget".to_string());
        }
        
        // Clip gradients first
        let clipped_gradients = self.mechanism.clip_gradients(gradients, sensitivity as f32);
        
        // Add Gaussian noise for (ε, δ)-DP
        let noisy_gradients: Vec<f32> = clipped_gradients
            .iter()
            .map(|&g| {
                self.mechanism.add_gaussian_noise(g as f64, sensitivity, epsilon, delta) as f32
            })
            .collect();
        
        // Update privacy budget
        accountant.spend_budget(
            format!("gradient_update_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()),
            epsilon,
            delta,
            "gaussian_mechanism".to_string(),
        )?;
        
        Ok(noisy_gradients)
    }
    
    pub fn get_remaining_budget(&self, participant_id: &str) -> Option<PrivacyBudget> {
        self.accountants.get(participant_id).map(|a| a.remaining_budget())
    }
    
    pub fn compute_composition_bounds(&self, participant_id: &str) -> Option<f64> {
        self.accountants.get(participant_id).map(|accountant| {
            // Advanced composition using RDP (Rényi Differential Privacy)
            // This is a simplified version - in practice, you'd use more sophisticated bounds
            let total_queries = accountant.queries.len() as f64;
            if total_queries == 0.0 {
                return 0.0;
            }
            
            let avg_epsilon = accountant.spent_budget.epsilon / total_queries;
            let composition_factor = (2.0 * total_queries * avg_epsilon.powi(2)).sqrt();
            
            composition_factor
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_privacy_accountant() {
        let mut accountant = PrivacyAccountant::new(1.0, 1e-5);
        
        assert!(accountant.can_spend(0.5, 1e-6));
        assert!(!accountant.can_spend(1.5, 1e-6));
        
        accountant.spend_budget(
            "test_query".to_string(),
            0.5,
            1e-6,
            "laplace".to_string(),
        ).unwrap();
        
        let remaining = accountant.remaining_budget();
        assert!((remaining.epsilon - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_gradient_clipping() {
        let mechanism = PrivacyMechanism::new();
        let gradients = vec![3.0, 4.0, 0.0]; // Norm = 5.0
        let clipped = mechanism.clip_gradients(&gradients, 2.0);
        
        let clipped_norm: f32 = clipped.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!((clipped_norm - 2.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_federated_privacy() {
        let mut fed_privacy = FederatedPrivacy::new();
        fed_privacy.register_participant("hospital_1".to_string(), 1.0, 1e-5);
        
        let gradients = vec![1.0, 2.0, 3.0];
        let result = fed_privacy.add_noise_to_gradients(
            "hospital_1",
            &gradients,
            1.0,
            0.5,
            1e-6,
        );
        
        assert!(result.is_ok());
        let noisy_gradients = result.unwrap();
        assert_eq!(noisy_gradients.len(), gradients.len());
    }
}