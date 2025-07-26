use candid::{CandidType, Deserialize, Principal};
use ic_cdk_macros::*;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;
use rand::Rng;
use sha2::{Digest, Sha256};

#[derive(CandidType, Deserialize, Serialize, Clone, Debug)]
pub struct GradientUpdate {
    pub institution_id: String,
    pub model_version: String,
    pub gradients: Vec<f32>,
    pub sample_count: u32,
    pub privacy_budget: f64,
    pub timestamp: u64,
    pub signature: Vec<u8>,
}

#[derive(CandidType, Deserialize, Serialize, Clone, Debug)]
pub struct AggregatedModel {
    pub version: String,
    pub weights: Vec<f32>,
    pub participating_institutions: Vec<String>,
    pub privacy_spent: f64,
    pub aggregation_round: u64,
    pub threshold_signature: Vec<u8>,
}

#[derive(CandidType, Deserialize, Serialize, Clone, Debug)]
pub struct InstitutionMetrics {
    pub institution_id: String,
    pub total_contributions: u32,
    pub privacy_budget_used: f64,
    pub last_update: u64,
    pub reputation_score: f64,
}

#[derive(CandidType, Deserialize, Serialize, Clone, Debug)]
pub struct FederatedRound {
    pub round_id: u64,
    pub status: RoundStatus,
    pub target_participants: u32,
    pub current_participants: u32,
    pub privacy_epsilon: f64,
    pub deadline: u64,
    pub updates: Vec<GradientUpdate>,
}

#[derive(CandidType, Deserialize, Serialize, Clone, Debug)]
pub enum RoundStatus {
    Open,
    Aggregating,
    Completed,
    Failed,
}

thread_local! {
    static CURRENT_ROUND: RefCell<Option<FederatedRound>> = RefCell::new(None);
    static INSTITUTION_REGISTRY: RefCell<HashMap<String, InstitutionMetrics>> = RefCell::new(HashMap::new());
    static MODEL_HISTORY: RefCell<Vec<AggregatedModel>> = RefCell::new(Vec::new());
    static PRIVACY_ACCOUNTANT: RefCell<HashMap<String, f64>> = RefCell::new(HashMap::new());
}

const MAX_PRIVACY_BUDGET: f64 = 10.0;
const MIN_PARTICIPANTS: u32 = 3;

#[init]
fn init() {
    ic_cdk::println!("Federated Aggregator Canister initialized");
    
    // Initialize first federated learning round
    start_new_round(MIN_PARTICIPANTS, 1.0);
}

#[update]
fn register_institution(institution_id: String) -> Result<String, String> {
    if institution_id.is_empty() {
        return Err("Institution ID cannot be empty".to_string());
    }
    
    INSTITUTION_REGISTRY.with(|registry| {
        let mut reg = registry.borrow_mut();
        if reg.contains_key(&institution_id) {
            return Err("Institution already registered".to_string());
        }
        
        let metrics = InstitutionMetrics {
            institution_id: institution_id.clone(),
            total_contributions: 0,
            privacy_budget_used: 0.0,
            last_update: ic_cdk::api::time(),
            reputation_score: 1.0,
        };
        
        reg.insert(institution_id.clone(), metrics);
        Ok(format!("Institution {} registered successfully", institution_id))
    })
}

#[update]
fn submit_gradient_update(update: GradientUpdate) -> Result<String, String> {
    // Verify institution is registered
    let institution_exists = INSTITUTION_REGISTRY.with(|registry| {
        registry.borrow().contains_key(&update.institution_id)
    });
    
    if !institution_exists {
        return Err("Institution not registered".to_string());
    }
    
    // Check privacy budget
    let privacy_available = PRIVACY_ACCOUNTANT.with(|accountant| {
        let acc = accountant.borrow();
        let used = acc.get(&update.institution_id).unwrap_or(&0.0);
        MAX_PRIVACY_BUDGET - used
    });
    
    if update.privacy_budget > privacy_available {
        return Err("Insufficient privacy budget".to_string());
    }
    
    // Verify gradient update signature (simplified)
    if !verify_gradient_signature(&update) {
        return Err("Invalid gradient signature".to_string());
    }
    
    // Add differential privacy noise
    let noisy_gradients = add_differential_privacy_noise(&update.gradients, update.privacy_budget);
    
    let mut noisy_update = update.clone();
    noisy_update.gradients = noisy_gradients;
    
    // Add to current round
    CURRENT_ROUND.with(|round| {
        let mut current = round.borrow_mut();
        if let Some(ref mut round_data) = *current {
            if matches!(round_data.status, RoundStatus::Open) {
                round_data.updates.push(noisy_update);
                round_data.current_participants += 1;
                
                // Update privacy accountant
                PRIVACY_ACCOUNTANT.with(|accountant| {
                    let mut acc = accountant.borrow_mut();
                    let current_used = acc.get(&update.institution_id).unwrap_or(&0.0);
                    acc.insert(update.institution_id.clone(), current_used + update.privacy_budget);
                });
                
                // Update institution metrics
                INSTITUTION_REGISTRY.with(|registry| {
                    let mut reg = registry.borrow_mut();
                    if let Some(metrics) = reg.get_mut(&update.institution_id) {
                        metrics.total_contributions += 1;
                        metrics.privacy_budget_used += update.privacy_budget;
                        metrics.last_update = ic_cdk::api::time();
                    }
                });
                
                // Check if we can start aggregation
                if round_data.current_participants >= round_data.target_participants {
                    round_data.status = RoundStatus::Aggregating;
                    ic_cdk::spawn(async {
                        if let Err(e) = perform_aggregation().await {
                            ic_cdk::println!("Aggregation failed: {}", e);
                        }
                    });
                }
                
                Ok("Gradient update submitted successfully".to_string())
            } else {
                Err("Current round is not accepting updates".to_string())
            }
        } else {
            Err("No active round".to_string())
        }
    })
}

fn add_differential_privacy_noise(gradients: &[f32], epsilon: f64) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let sensitivity = 1.0; // L2 sensitivity for gradient clipping
    let scale = sensitivity / epsilon;
    
    gradients.iter().map(|&grad| {
        let noise: f64 = rng.gen_range(-scale..scale);
        grad + noise as f32
    }).collect()
}

async fn perform_aggregation() -> Result<(), String> {
    let updates = CURRENT_ROUND.with(|round| {
        round.borrow().as_ref().map(|r| r.updates.clone()).unwrap_or_default()
    });
    
    if updates.is_empty() {
        return Err("No updates to aggregate".to_string());
    }
    
    // Federated averaging with differential privacy
    let aggregated_weights = federated_average(&updates)?;
    
    // Create new model version
    let new_version = format!("v{}", ic_cdk::api::time());
    let participating_institutions: Vec<String> = updates.iter()
        .map(|u| u.institution_id.clone())
        .collect();
    
    let total_privacy_spent: f64 = updates.iter()
        .map(|u| u.privacy_budget)
        .sum();
    
    let aggregated_model = AggregatedModel {
        version: new_version.clone(),
        weights: aggregated_weights,
        participating_institutions,
        privacy_spent: total_privacy_spent,
        aggregation_round: ic_cdk::api::time(),
        threshold_signature: generate_threshold_signature(&new_version),
    };
    
    // Store in model history
    MODEL_HISTORY.with(|history| {
        history.borrow_mut().push(aggregated_model);
    });
    
    // Mark round as completed and start new round
    CURRENT_ROUND.with(|round| {
        if let Some(ref mut round_data) = *round.borrow_mut() {
            round_data.status = RoundStatus::Completed;
        }
    });
    
    // Start next round
    start_new_round(MIN_PARTICIPANTS, 1.0);
    
    ic_cdk::println!("Aggregation completed for model version: {}", new_version);
    Ok(())
}

fn federated_average(updates: &[GradientUpdate]) -> Result<Vec<f32>, String> {
    if updates.is_empty() {
        return Err("No updates to average".to_string());
    }
    
    let gradient_size = updates[0].gradients.len();
    let mut averaged_gradients = vec![0.0f32; gradient_size];
    let mut total_samples = 0u32;
    
    // Weighted average by sample count
    for update in updates {
        if update.gradients.len() != gradient_size {
            return Err("Gradient size mismatch".to_string());
        }
        
        for (i, &gradient) in update.gradients.iter().enumerate() {
            averaged_gradients[i] += gradient * update.sample_count as f32;
        }
        total_samples += update.sample_count;
    }
    
    // Normalize by total samples
    for gradient in &mut averaged_gradients {
        *gradient /= total_samples as f32;
    }
    
    Ok(averaged_gradients)
}

fn verify_gradient_signature(update: &GradientUpdate) -> bool {
    // Simplified signature verification
    // In production, this would verify the institution's signature
    !update.signature.is_empty()
}

fn generate_threshold_signature(version: &str) -> Vec<u8> {
    // Simplified threshold signature generation
    // In production, this would use proper threshold cryptography
    let mut hasher = Sha256::new();
    hasher.update(version.as_bytes());
    hasher.finalize().to_vec()
}

fn start_new_round(target_participants: u32, privacy_epsilon: f64) {
    let round = FederatedRound {
        round_id: ic_cdk::api::time(),
        status: RoundStatus::Open,
        target_participants,
        current_participants: 0,
        privacy_epsilon,
        deadline: ic_cdk::api::time() + 3600_000_000_000, // 1 hour in nanoseconds
        updates: Vec::new(),
    };
    
    CURRENT_ROUND.with(|current| {
        *current.borrow_mut() = Some(round);
    });
    
    ic_cdk::println!("New federated learning round started");
}

#[query]
fn get_current_round() -> Option<FederatedRound> {
    CURRENT_ROUND.with(|round| round.borrow().clone())
}

#[query]
fn get_institution_metrics(institution_id: String) -> Option<InstitutionMetrics> {
    INSTITUTION_REGISTRY.with(|registry| {
        registry.borrow().get(&institution_id).cloned()
    })
}

#[query]
fn get_latest_model() -> Option<AggregatedModel> {
    MODEL_HISTORY.with(|history| {
        history.borrow().last().cloned()
    })
}

#[query]
fn get_privacy_budget(institution_id: String) -> f64 {
    PRIVACY_ACCOUNTANT.with(|accountant| {
        let used = accountant.borrow().get(&institution_id).unwrap_or(&0.0);
        MAX_PRIVACY_BUDGET - used
    })
}

#[query]
fn get_aggregator_status() -> HashMap<String, String> {
    let mut status = HashMap::new();
    
    let round_status = CURRENT_ROUND.with(|round| {
        round.borrow().as_ref().map(|r| format!("{:?}", r.status)).unwrap_or("None".to_string())
    });
    
    let total_institutions = INSTITUTION_REGISTRY.with(|registry| {
        registry.borrow().len().to_string()
    });
    
    let total_models = MODEL_HISTORY.with(|history| {
        history.borrow().len().to_string()
    });
    
    status.insert("current_round_status".to_string(), round_status);
    status.insert("registered_institutions".to_string(), total_institutions);
    status.insert("aggregated_models".to_string(), total_models);
    
    status
}

// Export Candid interface
ic_cdk::export_candid!();