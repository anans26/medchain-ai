use candid::CandidType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use differential_privacy::DifferentialPrivacy;

pub mod compression;
pub mod aggregation;
pub mod optimization;
pub mod communication;

// Core federated learning types
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct ModelUpdate {
    pub client_id: String,
    pub round: u64,
    pub gradients: Vec<f64>,
    pub weights: Vec<f64>,
    pub loss: f64,
    pub accuracy: f64,
    pub data_size: usize,
    pub computation_time: f64,
    pub communication_cost: f64,
    pub privacy_budget_used: f64,
    pub compressed: bool,
    pub compression_ratio: Option<f64>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct GlobalModel {
    pub round: u64,
    pub weights: Vec<f64>,
    pub global_loss: f64,
    pub global_accuracy: f64,
    pub participating_clients: Vec<String>,
    pub convergence_metrics: ConvergenceMetrics,
    pub privacy_metrics: PrivacyMetrics,
    pub communication_metrics: CommunicationMetrics,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct ConvergenceMetrics {
    pub gradient_norm: f64,
    pub weight_change_norm: f64,
    pub loss_improvement: f64,
    pub accuracy_improvement: f64,
    pub convergence_rate: f64,
    pub stability_score: f64,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct PrivacyMetrics {
    pub total_epsilon_used: f64,
    pub total_delta_used: f64,
    pub privacy_loss_per_client: HashMap<String, f64>,
    pub differential_privacy_guarantee: f64,
    pub membership_inference_resistance: f64,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct CommunicationMetrics {
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub compression_savings: f64,
    pub communication_rounds: u64,
    pub average_round_time: f64,
    pub bandwidth_efficiency: f64,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct FederatedLearningConfig {
    pub algorithm: FLAlgorithm,
    pub aggregation_method: AggregationMethod,
    pub compression_method: CompressionMethod,
    pub privacy_method: PrivacyMethod,
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub local_epochs: u32,
    pub batch_size: u32,
    pub client_fraction: f64,
    pub min_clients: u32,
    pub max_rounds: u32,
    pub convergence_threshold: f64,
    pub privacy_budget: PrivacyBudget,
    pub communication_budget: CommunicationBudget,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum FLAlgorithm {
    FedAvg,
    FedProx { mu: f64 },
    FedAdam { beta1: f64, beta2: f64 },
    FedAvgM { momentum: f64 },
    FedDyn { alpha: f64 },
    FedACG { lookahead_steps: u32 },
    SCAFFOLD,
    FedNova,
    FedOpt,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum AggregationMethod {
    WeightedAverage,
    FedAvg,
    Krum { byzantine_clients: u32 },
    TrimmedMean { trim_ratio: f64 },
    Median,
    FoolsGold,
    MultiKrum { m: u32 },
    Bulyan { f: u32 },
    SignSGD,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum CompressionMethod {
    None,
    Quantization { bits: u8 },
    Sparsification { sparsity_ratio: f64 },
    TopK { k: u32 },
    RandomK { k: u32 },
    SignSGD,
    TernGrad,
    QSGD { levels: u32 },
    DeepGradientCompression { compression_ratio: f64 },
    FedPAQ { quantization_levels: u32 },
    AdaptiveCompression { target_ratio: f64 },
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum PrivacyMethod {
    None,
    DifferentialPrivacy { epsilon: f64, delta: f64 },
    LocalDifferentialPrivacy { epsilon: f64 },
    SecureAggregation,
    HomomorphicEncryption,
    MultiPartyComputation,
    TrustedExecutionEnvironment,
    GradientObfuscation { noise_scale: f64 },
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct PrivacyBudget {
    pub total_epsilon: f64,
    pub total_delta: f64,
    pub per_round_epsilon: f64,
    pub per_client_epsilon: f64,
    pub composition_method: CompositionMethod,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum CompositionMethod {
    Basic,
    Advanced,
    RenyiDP { alpha: f64 },
    ZeroConcentratedDP,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct CommunicationBudget {
    pub max_bytes_per_round: u64,
    pub max_total_bytes: u64,
    pub target_compression_ratio: f64,
    pub adaptive_compression: bool,
}

// Main federated learning coordinator
pub struct FederatedLearningCoordinator {
    config: FederatedLearningConfig,
    global_model: GlobalModel,
    client_updates: HashMap<String, ModelUpdate>,
    round_history: Vec<GlobalModel>,
    privacy_engine: DifferentialPrivacy,
    compression_engine: CompressionEngine,
    aggregation_engine: AggregationEngine,
    optimization_engine: OptimizationEngine,
}

impl FederatedLearningCoordinator {
    pub fn new(config: FederatedLearningConfig) -> Self {
        let initial_weights = vec![0.0; 1000]; // Placeholder size
        
        let global_model = GlobalModel {
            round: 0,
            weights: initial_weights,
            global_loss: f64::INFINITY,
            global_accuracy: 0.0,
            participating_clients: Vec::new(),
            convergence_metrics: ConvergenceMetrics {
                gradient_norm: 0.0,
                weight_change_norm: 0.0,
                loss_improvement: 0.0,
                accuracy_improvement: 0.0,
                convergence_rate: 0.0,
                stability_score: 0.0,
            },
            privacy_metrics: PrivacyMetrics {
                total_epsilon_used: 0.0,
                total_delta_used: 0.0,
                privacy_loss_per_client: HashMap::new(),
                differential_privacy_guarantee: 0.0,
                membership_inference_resistance: 0.0,
            },
            communication_metrics: CommunicationMetrics {
                total_bytes_sent: 0,
                total_bytes_received: 0,
                compression_savings: 0.0,
                communication_rounds: 0,
                average_round_time: 0.0,
                bandwidth_efficiency: 0.0,
            },
        };

        FederatedLearningCoordinator {
            config,
            global_model,
            client_updates: HashMap::new(),
            round_history: Vec::new(),
            privacy_engine: DifferentialPrivacy::new(),
            compression_engine: CompressionEngine::new(),
            aggregation_engine: AggregationEngine::new(),
            optimization_engine: OptimizationEngine::new(),
        }
    }

    // Core federated learning round
    pub fn execute_round(&mut self, client_updates: Vec<ModelUpdate>) -> Result<GlobalModel, String> {
        // 1. Validate and filter client updates
        let valid_updates = self.validate_client_updates(client_updates)?;
        
        // 2. Apply privacy mechanisms
        let private_updates = self.apply_privacy_mechanisms(valid_updates)?;
        
        // 3. Decompress updates if needed
        let decompressed_updates = self.decompress_updates(private_updates)?;
        
        // 4. Aggregate updates using selected method
        let aggregated_weights = self.aggregate_updates(&decompressed_updates)?;
        
        // 5. Apply optimization algorithm
        let optimized_weights = self.apply_optimization(aggregated_weights)?;
        
        // 6. Update global model
        self.update_global_model(optimized_weights, &decompressed_updates)?;
        
        // 7. Compute convergence and privacy metrics
        self.compute_metrics(&decompressed_updates)?;
        
        // 8. Store round history
        self.round_history.push(self.global_model.clone());
        
        Ok(self.global_model.clone())
    }

    fn validate_client_updates(&self, updates: Vec<ModelUpdate>) -> Result<Vec<ModelUpdate>, String> {
        let mut valid_updates = Vec::new();
        
        for update in updates {
            // Check if client is authorized
            if !self.is_client_authorized(&update.client_id) {
                continue;
            }
            
            // Check if update is from current round
            if update.round != self.global_model.round {
                continue;
            }
            
            // Check gradient bounds (Byzantine fault tolerance)
            if self.is_gradient_valid(&update.gradients) {
                valid_updates.push(update);
            }
        }
        
        if valid_updates.len() < self.config.min_clients as usize {
            return Err("Insufficient valid client updates".to_string());
        }
        
        Ok(valid_updates)
    }

    fn apply_privacy_mechanisms(&mut self, updates: Vec<ModelUpdate>) -> Result<Vec<ModelUpdate>, String> {
        match &self.config.privacy_method {
            PrivacyMethod::None => Ok(updates),
            PrivacyMethod::DifferentialPrivacy { epsilon, delta } => {
                self.apply_differential_privacy(updates, *epsilon, *delta)
            }
            PrivacyMethod::LocalDifferentialPrivacy { epsilon } => {
                self.apply_local_differential_privacy(updates, *epsilon)
            }
            PrivacyMethod::SecureAggregation => {
                self.apply_secure_aggregation(updates)
            }
            PrivacyMethod::GradientObfuscation { noise_scale } => {
                self.apply_gradient_obfuscation(updates, *noise_scale)
            }
            _ => Err("Privacy method not implemented".to_string()),
        }
    }

    fn apply_differential_privacy(&mut self, mut updates: Vec<ModelUpdate>, epsilon: f64, delta: f64) -> Result<Vec<ModelUpdate>, String> {
        let sensitivity = self.compute_gradient_sensitivity(&updates);
        
        for update in &mut updates {
            // Add calibrated noise to gradients
            let noisy_gradients: Vec<f64> = update.gradients
                .iter()
                .map(|&gradient| {
                    gradient + self.privacy_engine.add_gaussian_noise(sensitivity, epsilon, delta)
                })
                .collect();
            
            update.gradients = noisy_gradients;
            update.privacy_budget_used = epsilon;
        }
        
        Ok(updates)
    }

    fn apply_local_differential_privacy(&mut self, mut updates: Vec<ModelUpdate>, epsilon: f64) -> Result<Vec<ModelUpdate>, String> {
        // Each client applies local DP before sending updates
        for update in &mut updates {
            let noise_scale = 2.0 / epsilon; // Laplace mechanism
            
            let noisy_gradients: Vec<f64> = update.gradients
                .iter()
                .map(|&gradient| {
                    gradient + self.sample_laplace_noise(0.0, noise_scale)
                })
                .collect();
            
            update.gradients = noisy_gradients;
            update.privacy_budget_used = epsilon;
        }
        
        Ok(updates)
    }

    fn apply_secure_aggregation(&self, updates: Vec<ModelUpdate>) -> Result<Vec<ModelUpdate>, String> {
        // Simplified secure aggregation - in practice would use cryptographic protocols
        // This is a placeholder for the actual secure aggregation implementation
        Ok(updates)
    }

    fn apply_gradient_obfuscation(&self, mut updates: Vec<ModelUpdate>, noise_scale: f64) -> Result<Vec<ModelUpdate>, String> {
        for update in &mut updates {
            let obfuscated_gradients: Vec<f64> = update.gradients
                .iter()
                .map(|&gradient| {
                    gradient + self.sample_gaussian_noise(0.0, noise_scale)
                })
                .collect();
            
            update.gradients = obfuscated_gradients;
        }
        
        Ok(updates)
    }

    fn decompress_updates(&self, updates: Vec<ModelUpdate>) -> Result<Vec<ModelUpdate>, String> {
        match &self.config.compression_method {
            CompressionMethod::None => Ok(updates),
            CompressionMethod::Quantization { bits } => {
                self.compression_engine.dequantize_updates(updates, *bits)
            }
            CompressionMethod::Sparsification { sparsity_ratio } => {
                self.compression_engine.desparsify_updates(updates, *sparsity_ratio)
            }
            CompressionMethod::TopK { k } => {
                self.compression_engine.decompress_topk_updates(updates, *k)
            }
            _ => Err("Compression method not implemented".to_string()),
        }
    }

    fn aggregate_updates(&self, updates: &[ModelUpdate]) -> Result<Vec<f64>, String> {
        match &self.config.aggregation_method {
            AggregationMethod::WeightedAverage => {
                self.aggregation_engine.weighted_average(updates)
            }
            AggregationMethod::FedAvg => {
                self.aggregation_engine.federated_averaging(updates)
            }
            AggregationMethod::Krum { byzantine_clients } => {
                self.aggregation_engine.krum_aggregation(updates, *byzantine_clients)
            }
            AggregationMethod::TrimmedMean { trim_ratio } => {
                self.aggregation_engine.trimmed_mean_aggregation(updates, *trim_ratio)
            }
            AggregationMethod::Median => {
                self.aggregation_engine.median_aggregation(updates)
            }
            _ => Err("Aggregation method not implemented".to_string()),
        }
    }

    fn apply_optimization(&mut self, weights: Vec<f64>) -> Result<Vec<f64>, String> {
        match &self.config.algorithm {
            FLAlgorithm::FedAvg => Ok(weights),
            FLAlgorithm::FedProx { mu } => {
                self.optimization_engine.fedprox_optimization(weights, *mu, &self.global_model.weights)
            }
            FLAlgorithm::FedAdam { beta1, beta2 } => {
                self.optimization_engine.fedadam_optimization(weights, *beta1, *beta2, self.config.learning_rate)
            }
            FLAlgorithm::FedAvgM { momentum } => {
                self.optimization_engine.fedavgm_optimization(weights, *momentum, &self.global_model.weights)
            }
            FLAlgorithm::SCAFFOLD => {
                self.optimization_engine.scaffold_optimization(weights, &self.global_model.weights)
            }
            _ => Err("Optimization algorithm not implemented".to_string()),
        }
    }

    fn update_global_model(&mut self, new_weights: Vec<f64>, updates: &[ModelUpdate]) -> Result<(), String> {
        let previous_weights = self.global_model.weights.clone();
        
        self.global_model.round += 1;
        self.global_model.weights = new_weights;
        self.global_model.participating_clients = updates.iter().map(|u| u.client_id.clone()).collect();
        
        // Compute global metrics
        self.global_model.global_loss = self.compute_weighted_average_loss(updates);
        self.global_model.global_accuracy = self.compute_weighted_average_accuracy(updates);
        
        // Update convergence metrics
        self.global_model.convergence_metrics.weight_change_norm = 
            self.compute_l2_norm_difference(&self.global_model.weights, &previous_weights);
        
        Ok(())
    }

    fn compute_metrics(&mut self, updates: &[ModelUpdate]) -> Result<(), String> {
        // Update privacy metrics
        let total_epsilon: f64 = updates.iter().map(|u| u.privacy_budget_used).sum();
        self.global_model.privacy_metrics.total_epsilon_used += total_epsilon;
        
        for update in updates {
            *self.global_model.privacy_metrics.privacy_loss_per_client
                .entry(update.client_id.clone())
                .or_insert(0.0) += update.privacy_budget_used;
        }
        
        // Update communication metrics
        let total_communication_cost: f64 = updates.iter().map(|u| u.communication_cost).sum();
        self.global_model.communication_metrics.total_bytes_sent += total_communication_cost as u64;
        self.global_model.communication_metrics.communication_rounds += 1;
        
        // Compute compression savings
        let compressed_updates: Vec<&ModelUpdate> = updates.iter().filter(|u| u.compressed).collect();
        if !compressed_updates.is_empty() {
            let avg_compression_ratio: f64 = compressed_updates
                .iter()
                .filter_map(|u| u.compression_ratio)
                .sum::<f64>() / compressed_updates.len() as f64;
            
            self.global_model.communication_metrics.compression_savings = avg_compression_ratio;
        }
        
        Ok(())
    }

    // Helper methods
    fn is_client_authorized(&self, _client_id: &str) -> bool {
        // In practice, implement proper client authorization
        true
    }

    fn is_gradient_valid(&self, gradients: &[f64]) -> bool {
        // Check for NaN, infinity, or extremely large values
        gradients.iter().all(|&g| g.is_finite() && g.abs() < 1e6)
    }

    fn compute_gradient_sensitivity(&self, updates: &[ModelUpdate]) -> f64 {
        // Compute L2 sensitivity for differential privacy
        let mut max_norm = 0.0;
        
        for update in updates {
            let norm = self.compute_l2_norm(&update.gradients);
            if norm > max_norm {
                max_norm = norm;
            }
        }
        
        max_norm
    }

    fn compute_l2_norm(&self, vector: &[f64]) -> f64 {
        vector.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    fn compute_l2_norm_difference(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn compute_weighted_average_loss(&self, updates: &[ModelUpdate]) -> f64 {
        let total_weight: f64 = updates.iter().map(|u| u.data_size as f64).sum();
        if total_weight == 0.0 {
            return f64::INFINITY;
        }
        
        updates.iter()
            .map(|u| u.loss * (u.data_size as f64 / total_weight))
            .sum()
    }

    fn compute_weighted_average_accuracy(&self, updates: &[ModelUpdate]) -> f64 {
        let total_weight: f64 = updates.iter().map(|u| u.data_size as f64).sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        
        updates.iter()
            .map(|u| u.accuracy * (u.data_size as f64 / total_weight))
            .sum()
    }

    fn sample_laplace_noise(&self, mean: f64, scale: f64) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let u: f64 = rng.gen_range(-0.5..0.5);
        mean - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }

    fn sample_gaussian_noise(&self, mean: f64, std_dev: f64) -> f64 {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(mean, std_dev).unwrap();
        normal.sample(&mut rand::thread_rng())
    }

    // Public API methods
    pub fn get_global_model(&self) -> &GlobalModel {
        &self.global_model
    }

    pub fn get_round_history(&self) -> &[GlobalModel] {
        &self.round_history
    }

    pub fn is_converged(&self) -> bool {
        if self.global_model.round < 10 {
            return false;
        }
        
        // Check convergence based on loss improvement
        let recent_losses: Vec<f64> = self.round_history
            .iter()
            .rev()
            .take(5)
            .map(|m| m.global_loss)
            .collect();
        
        if recent_losses.len() < 5 {
            return false;
        }
        
        let loss_variance = self.compute_variance(&recent_losses);
        loss_variance < self.config.convergence_threshold
    }

    fn compute_variance(&self, values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance
    }

    pub fn get_privacy_report(&self) -> PrivacyReport {
        PrivacyReport {
            total_epsilon_used: self.global_model.privacy_metrics.total_epsilon_used,
            total_delta_used: self.global_model.privacy_metrics.total_delta_used,
            privacy_budget_remaining: self.config.privacy_budget.total_epsilon - self.global_model.privacy_metrics.total_epsilon_used,
            client_privacy_usage: self.global_model.privacy_metrics.privacy_loss_per_client.clone(),
            rounds_remaining: self.estimate_remaining_rounds(),
        }
    }

    fn estimate_remaining_rounds(&self) -> u32 {
        let epsilon_per_round = if self.global_model.round > 0 {
            self.global_model.privacy_metrics.total_epsilon_used / self.global_model.round as f64
        } else {
            self.config.privacy_budget.per_round_epsilon
        };
        
        let remaining_epsilon = self.config.privacy_budget.total_epsilon - self.global_model.privacy_metrics.total_epsilon_used;
        
        if epsilon_per_round > 0.0 {
            (remaining_epsilon / epsilon_per_round) as u32
        } else {
            self.config.max_rounds - self.global_model.round as u32
        }
    }
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct PrivacyReport {
    pub total_epsilon_used: f64,
    pub total_delta_used: f64,
    pub privacy_budget_remaining: f64,
    pub client_privacy_usage: HashMap<String, f64>,
    pub rounds_remaining: u32,
}

// Compression engine for communication efficiency
pub struct CompressionEngine;

impl CompressionEngine {
    pub fn new() -> Self {
        CompressionEngine
    }

    pub fn dequantize_updates(&self, mut updates: Vec<ModelUpdate>, bits: u8) -> Result<Vec<ModelUpdate>, String> {
        let levels = 2_u32.pow(bits as u32) as f64;
        
        for update in &mut updates {
            update.gradients = update.gradients
                .iter()
                .map(|&quantized| {
                    // Dequantize from [0, levels-1] back to original range
                    let normalized = quantized / (levels - 1.0);
                    // Assuming original range was [-1, 1] for simplicity
                    2.0 * normalized - 1.0
                })
                .collect();
        }
        
        Ok(updates)
    }

    pub fn desparsify_updates(&self, updates: Vec<ModelUpdate>, _sparsity_ratio: f64) -> Result<Vec<ModelUpdate>, String> {
        // For sparsification, we typically just keep the sparse representation
        // The aggregation engine handles sparse vectors appropriately
        Ok(updates)
    }

    pub fn decompress_topk_updates(&self, updates: Vec<ModelUpdate>, _k: u32) -> Result<Vec<ModelUpdate>, String> {
        // TopK compression keeps only the top-k largest gradients
        // The rest are implicitly zero
        Ok(updates)
    }
}

// Aggregation engine for robust model updates
pub struct AggregationEngine;

impl AggregationEngine {
    pub fn new() -> Self {
        AggregationEngine
    }

    pub fn weighted_average(&self, updates: &[ModelUpdate]) -> Result<Vec<f64>, String> {
        if updates.is_empty() {
            return Err("No updates to aggregate".to_string());
        }

        let total_weight: f64 = updates.iter().map(|u| u.data_size as f64).sum();
        let gradient_size = updates[0].gradients.len();
        
        let mut aggregated = vec![0.0; gradient_size];
        
        for update in updates {
            let weight = update.data_size as f64 / total_weight;
            for (i, &gradient) in update.gradients.iter().enumerate() {
                aggregated[i] += weight * gradient;
            }
        }
        
        Ok(aggregated)
    }

    pub fn federated_averaging(&self, updates: &[ModelUpdate]) -> Result<Vec<f64>, String> {
        // FedAvg is essentially weighted averaging by data size
        self.weighted_average(updates)
    }

    pub fn krum_aggregation(&self, updates: &[ModelUpdate], byzantine_clients: u32) -> Result<Vec<f64>, String> {
        if updates.len() <= byzantine_clients as usize {
            return Err("Too many Byzantine clients".to_string());
        }

        // Compute pairwise distances
        let mut distances = Vec::new();
        for (i, update_i) in updates.iter().enumerate() {
            let mut client_distances = Vec::new();
            for (j, update_j) in updates.iter().enumerate() {
                if i != j {
                    let distance = self.compute_euclidean_distance(&update_i.gradients, &update_j.gradients);
                    client_distances.push(distance);
                }
            }
            client_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Sum of distances to closest n-f-2 clients
            let n = updates.len();
            let f = byzantine_clients as usize;
            let sum_distance: f64 = client_distances.iter().take(n - f - 2).sum();
            distances.push((i, sum_distance));
        }
        
        // Select client with minimum sum of distances
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let selected_client = distances[0].0;
        
        Ok(updates[selected_client].gradients.clone())
    }

    pub fn trimmed_mean_aggregation(&self, updates: &[ModelUpdate], trim_ratio: f64) -> Result<Vec<f64>, String> {
        if updates.is_empty() {
            return Err("No updates to aggregate".to_string());
        }

        let gradient_size = updates[0].gradients.len();
        let mut aggregated = vec![0.0; gradient_size];
        let trim_count = ((updates.len() as f64 * trim_ratio) / 2.0).floor() as usize;
        
        for i in 0..gradient_size {
            let mut values: Vec<f64> = updates.iter().map(|u| u.gradients[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Remove extreme values
            let trimmed_values = &values[trim_count..values.len() - trim_count];
            aggregated[i] = trimmed_values.iter().sum::<f64>() / trimmed_values.len() as f64;
        }
        
        Ok(aggregated)
    }

    pub fn median_aggregation(&self, updates: &[ModelUpdate]) -> Result<Vec<f64>, String> {
        if updates.is_empty() {
            return Err("No updates to aggregate".to_string());
        }

        let gradient_size = updates[0].gradients.len();
        let mut aggregated = vec![0.0; gradient_size];
        
        for i in 0..gradient_size {
            let mut values: Vec<f64> = updates.iter().map(|u| u.gradients[i]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let median = if values.len() % 2 == 0 {
                (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
            } else {
                values[values.len() / 2]
            };
            
            aggregated[i] = median;
        }
        
        Ok(aggregated)
    }

    fn compute_euclidean_distance(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// Optimization engine for advanced FL algorithms
pub struct OptimizationEngine {
    momentum_buffer: HashMap<String, Vec<f64>>,
    adam_m: Vec<f64>,
    adam_v: Vec<f64>,
    adam_t: u64,
}

impl OptimizationEngine {
    pub fn new() -> Self {
        OptimizationEngine {
            momentum_buffer: HashMap::new(),
            adam_m: Vec::new(),
            adam_v: Vec::new(),
            adam_t: 0,
        }
    }

    pub fn fedprox_optimization(&self, weights: Vec<f64>, mu: f64, global_weights: &[f64]) -> Result<Vec<f64>, String> {
        // FedProx adds a proximal term to prevent client drift
        let mut optimized_weights = weights;
        
        for (i, weight) in optimized_weights.iter_mut().enumerate() {
            if i < global_weights.len() {
                *weight = *weight - mu * (*weight - global_weights[i]);
            }
        }
        
        Ok(optimized_weights)
    }

    pub fn fedadam_optimization(&mut self, weights: Vec<f64>, beta1: f64, beta2: f64, learning_rate: f64) -> Result<Vec<f64>, String> {
        // Initialize Adam buffers if needed
        if self.adam_m.len() != weights.len() {
            self.adam_m = vec![0.0; weights.len()];
            self.adam_v = vec![0.0; weights.len()];
        }
        
        self.adam_t += 1;
        let mut optimized_weights = weights.clone();
        
        for i in 0..weights.len() {
            let gradient = weights[i]; // Assuming weights are actually gradients here
            
            // Update biased first moment estimate
            self.adam_m[i] = beta1 * self.adam_m[i] + (1.0 - beta1) * gradient;
            
            // Update biased second raw moment estimate
            self.adam_v[i] = beta2 * self.adam_v[i] + (1.0 - beta2) * gradient * gradient;
            
            // Compute bias-corrected first moment estimate
            let m_hat = self.adam_m[i] / (1.0 - beta1.powi(self.adam_t as i32));
            
            // Compute bias-corrected second raw moment estimate
            let v_hat = self.adam_v[i] / (1.0 - beta2.powi(self.adam_t as i32));
            
            // Update weights
            optimized_weights[i] = weights[i] - learning_rate * m_hat / (v_hat.sqrt() + 1e-8);
        }
        
        Ok(optimized_weights)
    }

    pub fn fedavgm_optimization(&mut self, weights: Vec<f64>, momentum: f64, global_weights: &[f64]) -> Result<Vec<f64>, String> {
        let key = "global".to_string();
        
        // Initialize momentum buffer if needed
        if !self.momentum_buffer.contains_key(&key) {
            self.momentum_buffer.insert(key.clone(), vec![0.0; weights.len()]);
        }
        
        let momentum_buffer = self.momentum_buffer.get_mut(&key).unwrap();
        let mut optimized_weights = weights;
        
        for i in 0..optimized_weights.len() {
            let gradient = if i < global_weights.len() {
                optimized_weights[i] - global_weights[i]
            } else {
                optimized_weights[i]
            };
            
            momentum_buffer[i] = momentum * momentum_buffer[i] + gradient;
            optimized_weights[i] = global_weights[i] + momentum_buffer[i];
        }
        
        Ok(optimized_weights)
    }

    pub fn scaffold_optimization(&self, weights: Vec<f64>, _global_weights: &[f64]) -> Result<Vec<f64>, String> {
        // SCAFFOLD requires client-side control variates
        // This is a simplified version - full implementation requires more state
        Ok(weights)
    }
}

// Performance benchmarking and analysis
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct PerformanceBenchmark {
    pub algorithm: String,
    pub dataset_size: usize,
    pub num_clients: u32,
    pub rounds_to_convergence: u32,
    pub final_accuracy: f64,
    pub total_communication_cost: u64,
    pub total_computation_time: f64,
    pub privacy_budget_used: f64,
    pub compression_ratio: f64,
    pub memory_usage: u64,
    pub energy_consumption: f64,
}

pub fn benchmark_federated_algorithms(
    algorithms: Vec<FLAlgorithm>,
    dataset_size: usize,
    num_clients: u32,
) -> Vec<PerformanceBenchmark> {
    let mut benchmarks = Vec::new();
    
    for algorithm in algorithms {
        let config = FederatedLearningConfig {
            algorithm: algorithm.clone(),
            aggregation_method: AggregationMethod::FedAvg,
            compression_method: CompressionMethod::Quantization { bits: 8 },
            privacy_method: PrivacyMethod::DifferentialPrivacy { epsilon: 1.0, delta: 1e-5 },
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 1e-4,
            local_epochs: 5,
            batch_size: 32,
            client_fraction: 0.1,
            min_clients: 10,
            max_rounds: 100,
            convergence_threshold: 1e-4,
            privacy_budget: PrivacyBudget {
                total_epsilon: 10.0,
                total_delta: 1e-3,
                per_round_epsilon: 0.1,
                per_client_epsilon: 0.01,
                composition_method: CompositionMethod::Advanced,
            },
            communication_budget: CommunicationBudget {
                max_bytes_per_round: 1_000_000,
                max_total_bytes: 100_000_000,
                target_compression_ratio: 0.1,
                adaptive_compression: true,
            },
        };
        
        let benchmark = simulate_federated_learning(config, dataset_size, num_clients);
        benchmarks.push(benchmark);
    }
    
    benchmarks
}

fn simulate_federated_learning(
    config: FederatedLearningConfig,
    dataset_size: usize,
    num_clients: u32,
) -> PerformanceBenchmark {
    // This is a simplified simulation - in practice would run actual FL training
    let algorithm_name = format!("{:?}", config.algorithm);
    
    // Simulate performance based on algorithm characteristics
    let (rounds_to_convergence, final_accuracy, communication_cost) = match config.algorithm {
        FLAlgorithm::FedAvg => (50, 0.85, 1_000_000),
        FLAlgorithm::FedProx { .. } => (45, 0.87, 1_100_000),
        FLAlgorithm::FedAdam { .. } => (40, 0.89, 1_200_000),
        FLAlgorithm::FedAvgM { .. } => (42, 0.88, 1_050_000),
        FLAlgorithm::SCAFFOLD => (35, 0.91, 800_000),
        _ => (50, 0.85, 1_000_000),
    };
    
    PerformanceBenchmark {
        algorithm: algorithm_name,
        dataset_size,
        num_clients,
        rounds_to_convergence,
        final_accuracy,
        total_communication_cost: communication_cost,
        total_computation_time: rounds_to_convergence as f64 * 10.0, // 10 seconds per round
        privacy_budget_used: config.privacy_budget.total_epsilon * 0.8,
        compression_ratio: match config.compression_method {
            CompressionMethod::Quantization { bits } => 32.0 / bits as f64,
            CompressionMethod::Sparsification { sparsity_ratio } => 1.0 / (1.0 - sparsity_ratio),
            _ => 1.0,
        },
        memory_usage: dataset_size as u64 * 4, // 4 bytes per float
        energy_consumption: rounds_to_convergence as f64 * num_clients as f64 * 0.1, // 0.1 kWh per client per round
    }
}

// Cost analysis and optimization
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct CostAnalysis {
    pub communication_cost: f64,
    pub computation_cost: f64,
    pub storage_cost: f64,
    pub privacy_cost: f64,
    pub total_cost: f64,
    pub cost_per_accuracy_point: f64,
    pub cost_savings_vs_centralized: f64,
}

pub fn analyze_federated_learning_costs(
    benchmark: &PerformanceBenchmark,
    communication_cost_per_mb: f64,
    computation_cost_per_hour: f64,
    storage_cost_per_gb_month: f64,
) -> CostAnalysis {
    let communication_mb = benchmark.total_communication_cost as f64 / 1_000_000.0;
    let computation_hours = benchmark.total_computation_time / 3600.0;
    let storage_gb = benchmark.memory_usage as f64 / 1_000_000_000.0;
    
    let communication_cost = communication_mb * communication_cost_per_mb;
    let computation_cost = computation_hours * computation_cost_per_hour;
    let storage_cost = storage_gb * storage_cost_per_gb_month;
    
    // Privacy cost is estimated based on the complexity of privacy mechanisms
    let privacy_cost = benchmark.privacy_budget_used * 10.0; // $10 per epsilon unit
    
    let total_cost = communication_cost + computation_cost + storage_cost + privacy_cost;
    let cost_per_accuracy_point = total_cost / (benchmark.final_accuracy * 100.0);
    
    // Estimate cost savings vs centralized approach
    let centralized_communication_cost = communication_mb * 2.0; // 2x more communication
    let centralized_total_cost = centralized_communication_cost + computation_cost + storage_cost;
    let cost_savings_vs_centralized = (centralized_total_cost - total_cost) / centralized_total_cost * 100.0;
    
    CostAnalysis {
        communication_cost,
        computation_cost,
        storage_cost,
        privacy_cost,
        total_cost,
        cost_per_accuracy_point,
        cost_savings_vs_centralized,
    }
}

// Export main types and functions
pub use compression::*;
pub use aggregation::*;
pub use optimization::*;
pub use communication::*;