// Communication-efficient compression techniques for federated learning
// Based on research from "Deep Gradient Compression" and "QSGD" papers

use crate::*;
use std::collections::HashMap;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_time: f64,
    pub decompression_time: f64,
    pub accuracy_loss: f64,
}

// Quantization-based compression
pub struct QuantizationCompressor {
    pub bits: u8,
    pub stochastic: bool,
    pub level_mapping: HashMap<u32, f64>,
}

impl QuantizationCompressor {
    pub fn new(bits: u8, stochastic: bool) -> Self {
        QuantizationCompressor {
            bits,
            stochastic,
            level_mapping: HashMap::new(),
        }
    }

    // QSGD: Communication-Efficient SGD via Gradient Quantization
    pub fn qsgd_compress(&mut self, gradients: &[f64]) -> (Vec<u32>, f64, f64) {
        let levels = 2_u32.pow(self.bits as u32);
        let norm = self.compute_l2_norm(gradients);
        
        if norm == 0.0 {
            return (vec![0; gradients.len()], norm, 0.0);
        }
        
        let mut quantized = Vec::with_capacity(gradients.len());
        let mut total_error = 0.0;
        
        for &gradient in gradients {
            let normalized = gradient / norm;
            let abs_normalized = normalized.abs();
            
            let quantized_val = if self.stochastic {
                // Stochastic quantization
                let scaled = abs_normalized * (levels - 1) as f64;
                let floor_val = scaled.floor() as u32;
                let prob = scaled - floor_val as f64;
                
                if rand::random::<f64>() < prob {
                    floor_val + 1
                } else {
                    floor_val
                }
            } else {
                // Deterministic quantization
                (abs_normalized * (levels - 1) as f64).round() as u32
            };
            
            // Store sign information in MSB for signed quantization
            let signed_quantized = if gradient >= 0.0 {
                quantized_val
            } else {
                quantized_val | (1 << (self.bits - 1))
            };
            
            quantized.push(signed_quantized);
            
            // Calculate quantization error
            let dequantized = self.dequantize_single(signed_quantized, norm);
            total_error += (gradient - dequantized).powi(2);
        }
        
        (quantized, norm, total_error.sqrt())
    }

    pub fn qsgd_decompress(&self, quantized: &[u32], norm: f64) -> Vec<f64> {
        quantized.iter()
            .map(|&q| self.dequantize_single(q, norm))
            .collect()
    }

    fn dequantize_single(&self, quantized: u32, norm: f64) -> f64 {
        let levels = 2_u32.pow(self.bits as u32);
        let sign_mask = 1 << (self.bits - 1);
        
        let is_negative = (quantized & sign_mask) != 0;
        let magnitude = quantized & (sign_mask - 1);
        
        let normalized = magnitude as f64 / (levels - 1) as f64;
        let value = normalized * norm;
        
        if is_negative { -value } else { value }
    }

    // Adaptive quantization based on gradient statistics
    pub fn adaptive_quantize(&mut self, gradients: &[f64], target_error: f64) -> (Vec<u32>, CompressionStats) {
        let start_time = std::time::Instant::now();
        
        // Analyze gradient distribution
        let stats = self.analyze_gradient_distribution(gradients);
        
        // Adjust quantization levels based on distribution
        let optimal_bits = self.compute_optimal_bits(&stats, target_error);
        self.bits = optimal_bits;
        
        let (quantized, norm, error) = self.qsgd_compress(gradients);
        let compression_time = start_time.elapsed().as_secs_f64();
        
        let original_size = gradients.len() * 8; // 8 bytes per f64
        let compressed_size = quantized.len() * (self.bits as usize / 8).max(1) + 8; // +8 for norm
        
        let stats = CompressionStats {
            original_size,
            compressed_size,
            compression_ratio: original_size as f64 / compressed_size as f64,
            compression_time,
            decompression_time: 0.0,
            accuracy_loss: error,
        };
        
        (quantized, stats)
    }

    fn analyze_gradient_distribution(&self, gradients: &[f64]) -> GradientStats {
        let mut stats = GradientStats::new();
        
        for &grad in gradients {
            stats.update(grad);
        }
        
        stats.finalize();
        stats
    }

    fn compute_optimal_bits(&self, stats: &GradientStats, target_error: f64) -> u8 {
        // Use gradient statistics to determine optimal quantization bits
        let dynamic_range = stats.max - stats.min;
        let required_precision = target_error / dynamic_range;
        
        let optimal_bits = (-required_precision.log2()).ceil() as u8;
        optimal_bits.clamp(1, 16) // Reasonable bounds
    }

    fn compute_l2_norm(&self, vector: &[f64]) -> f64 {
        vector.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

// Sparsification-based compression
pub struct SparsificationCompressor {
    pub sparsity_ratio: f64,
    pub method: SparsificationMethod,
    pub momentum_buffer: HashMap<String, Vec<f64>>,
}

#[derive(Clone, Debug)]
pub enum SparsificationMethod {
    TopK,
    RandomK,
    ThresholdBased,
    AdaptiveThreshold,
}

impl SparsificationCompressor {
    pub fn new(sparsity_ratio: f64, method: SparsificationMethod) -> Self {
        SparsificationCompressor {
            sparsity_ratio,
            method,
            momentum_buffer: HashMap::new(),
        }
    }

    // Deep Gradient Compression with error feedback
    pub fn dgc_compress(&mut self, gradients: &[f64], client_id: &str) -> (SparseGradients, CompressionStats) {
        let start_time = std::time::Instant::now();
        
        // Get or initialize momentum buffer
        let momentum = self.momentum_buffer
            .entry(client_id.to_string())
            .or_insert_with(|| vec![0.0; gradients.len()]);
        
        // Add momentum to gradients (error feedback)
        let mut accumulated_gradients: Vec<f64> = gradients.iter()
            .zip(momentum.iter())
            .map(|(&grad, &mom)| grad + mom)
            .collect();
        
        // Apply sparsification
        let sparse_gradients = match self.method {
            SparsificationMethod::TopK => self.top_k_sparsify(&accumulated_gradients),
            SparsificationMethod::RandomK => self.random_k_sparsify(&accumulated_gradients),
            SparsificationMethod::ThresholdBased => self.threshold_sparsify(&accumulated_gradients),
            SparsificationMethod::AdaptiveThreshold => self.adaptive_threshold_sparsify(&accumulated_gradients),
        };
        
        // Update momentum buffer with residual (error feedback)
        for (i, &sparse_grad) in sparse_gradients.values.iter().enumerate() {
            let index = sparse_gradients.indices[i];
            momentum[index] = accumulated_gradients[index] - sparse_grad;
        }
        
        let compression_time = start_time.elapsed().as_secs_f64();
        
        let original_size = gradients.len() * 8;
        let compressed_size = sparse_gradients.indices.len() * 12; // 4 bytes index + 8 bytes value
        
        let stats = CompressionStats {
            original_size,
            compressed_size,
            compression_ratio: original_size as f64 / compressed_size as f64,
            compression_time,
            decompression_time: 0.0,
            accuracy_loss: self.compute_sparsification_error(gradients, &sparse_gradients),
        };
        
        (sparse_gradients, stats)
    }

    fn top_k_sparsify(&self, gradients: &[f64]) -> SparseGradients {
        let k = ((1.0 - self.sparsity_ratio) * gradients.len() as f64) as usize;
        
        // Create (index, value, magnitude) tuples
        let mut indexed_gradients: Vec<(usize, f64, f64)> = gradients.iter()
            .enumerate()
            .map(|(i, &val)| (i, val, val.abs()))
            .collect();
        
        // Sort by magnitude (descending)
        indexed_gradients.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        // Take top-k
        let top_k = indexed_gradients.into_iter().take(k);
        
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (index, value, _) in top_k {
            indices.push(index);
            values.push(value);
        }
        
        SparseGradients { indices, values }
    }

    fn random_k_sparsify(&self, gradients: &[f64]) -> SparseGradients {
        let k = ((1.0 - self.sparsity_ratio) * gradients.len() as f64) as usize;
        let mut rng = rand::thread_rng();
        
        // Randomly select k indices
        let mut indices: Vec<usize> = (0..gradients.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(k);
        
        let values: Vec<f64> = indices.iter().map(|&i| gradients[i]).collect();
        
        SparseGradients { indices, values }
    }

    fn threshold_sparsify(&self, gradients: &[f64]) -> SparseGradients {
        // Calculate threshold based on sparsity ratio
        let mut abs_gradients: Vec<f64> = gradients.iter().map(|&x| x.abs()).collect();
        abs_gradients.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let threshold_index = (self.sparsity_ratio * abs_gradients.len() as f64) as usize;
        let threshold = abs_gradients.get(threshold_index).unwrap_or(&0.0);
        
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (i, &grad) in gradients.iter().enumerate() {
            if grad.abs() >= *threshold {
                indices.push(i);
                values.push(grad);
            }
        }
        
        SparseGradients { indices, values }
    }

    fn adaptive_threshold_sparsify(&self, gradients: &[f64]) -> SparseGradients {
        // Adaptive threshold based on gradient statistics
        let mean_abs = gradients.iter().map(|&x| x.abs()).sum::<f64>() / gradients.len() as f64;
        let variance = gradients.iter()
            .map(|&x| (x.abs() - mean_abs).powi(2))
            .sum::<f64>() / gradients.len() as f64;
        let std_dev = variance.sqrt();
        
        let adaptive_threshold = mean_abs + 2.0 * std_dev; // 2-sigma threshold
        
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (i, &grad) in gradients.iter().enumerate() {
            if grad.abs() >= adaptive_threshold {
                indices.push(i);
                values.push(grad);
            }
        }
        
        // Ensure minimum sparsity
        if values.len() as f64 / gradients.len() as f64 > (1.0 - self.sparsity_ratio) {
            return self.top_k_sparsify(gradients);
        }
        
        SparseGradients { indices, values }
    }

    pub fn decompress(&self, sparse: &SparseGradients, original_size: usize) -> Vec<f64> {
        let start_time = std::time::Instant::now();
        
        let mut dense = vec![0.0; original_size];
        
        for (i, &index) in sparse.indices.iter().enumerate() {
            if index < original_size {
                dense[index] = sparse.values[i];
            }
        }
        
        dense
    }

    fn compute_sparsification_error(&self, original: &[f64], sparse: &SparseGradients) -> f64 {
        let reconstructed = self.decompress(sparse, original.len());
        
        original.iter()
            .zip(reconstructed.iter())
            .map(|(&orig, &recon)| (orig - recon).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// Hybrid compression combining multiple techniques
pub struct HybridCompressor {
    quantizer: QuantizationCompressor,
    sparsifier: SparsificationCompressor,
    compression_strategy: CompressionStrategy,
}

#[derive(Clone, Debug)]
pub enum CompressionStrategy {
    QuantizationFirst,
    SparsificationFirst,
    Adaptive,
    LayerWise,
}

impl HybridCompressor {
    pub fn new(
        bits: u8,
        sparsity_ratio: f64,
        strategy: CompressionStrategy,
    ) -> Self {
        HybridCompressor {
            quantizer: QuantizationCompressor::new(bits, true),
            sparsifier: SparsificationCompressor::new(sparsity_ratio, SparsificationMethod::TopK),
            compression_strategy: strategy,
        }
    }

    pub fn compress(&mut self, gradients: &[f64], client_id: &str) -> (HybridCompressedGradients, CompressionStats) {
        match self.compression_strategy {
            CompressionStrategy::QuantizationFirst => {
                self.quantization_first_compress(gradients, client_id)
            }
            CompressionStrategy::SparsificationFirst => {
                self.sparsification_first_compress(gradients, client_id)
            }
            CompressionStrategy::Adaptive => {
                self.adaptive_compress(gradients, client_id)
            }
            CompressionStrategy::LayerWise => {
                self.layer_wise_compress(gradients, client_id)
            }
        }
    }

    fn quantization_first_compress(&mut self, gradients: &[f64], client_id: &str) -> (HybridCompressedGradients, CompressionStats) {
        // First quantize, then sparsify
        let (quantized, norm, _) = self.quantizer.qsgd_compress(gradients);
        let dequantized = self.quantizer.qsgd_decompress(&quantized, norm);
        let (sparse, sparse_stats) = self.sparsifier.dgc_compress(&dequantized, client_id);
        
        let hybrid = HybridCompressedGradients {
            method: "quantization_first".to_string(),
            quantized_data: Some(quantized),
            sparse_data: Some(sparse),
            norm: Some(norm),
            metadata: HashMap::new(),
        };
        
        (hybrid, sparse_stats)
    }

    fn sparsification_first_compress(&mut self, gradients: &[f64], client_id: &str) -> (HybridCompressedGradients, CompressionStats) {
        // First sparsify, then quantize the sparse values
        let (sparse, _) = self.sparsifier.dgc_compress(gradients, client_id);
        let (quantized_values, norm, _) = self.quantizer.qsgd_compress(&sparse.values);
        
        let mut quantized_sparse = sparse.clone();
        quantized_sparse.values = self.quantizer.qsgd_decompress(&quantized_values, norm);
        
        let hybrid = HybridCompressedGradients {
            method: "sparsification_first".to_string(),
            quantized_data: Some(quantized_values),
            sparse_data: Some(quantized_sparse),
            norm: Some(norm),
            metadata: HashMap::new(),
        };
        
        let original_size = gradients.len() * 8;
        let compressed_size = quantized_values.len() * 4 + sparse.indices.len() * 4 + 8;
        
        let stats = CompressionStats {
            original_size,
            compressed_size,
            compression_ratio: original_size as f64 / compressed_size as f64,
            compression_time: 0.0,
            decompression_time: 0.0,
            accuracy_loss: 0.0,
        };
        
        (hybrid, stats)
    }

    fn adaptive_compress(&mut self, gradients: &[f64], client_id: &str) -> (HybridCompressedGradients, CompressionStats) {
        // Analyze gradient characteristics to choose optimal compression
        let sparsity = self.compute_natural_sparsity(gradients);
        let dynamic_range = self.compute_dynamic_range(gradients);
        
        if sparsity > 0.7 && dynamic_range > 100.0 {
            // High sparsity and large dynamic range: sparsify first
            self.sparsification_first_compress(gradients, client_id)
        } else if dynamic_range > 1000.0 {
            // Large dynamic range: quantize first
            self.quantization_first_compress(gradients, client_id)
        } else {
            // Default to quantization first
            self.quantization_first_compress(gradients, client_id)
        }
    }

    fn layer_wise_compress(&mut self, gradients: &[f64], client_id: &str) -> (HybridCompressedGradients, CompressionStats) {
        // Apply different compression strategies to different parts of the gradient
        // This is a simplified version - in practice would split by actual layer boundaries
        
        let mid_point = gradients.len() / 2;
        let first_half = &gradients[..mid_point];
        let second_half = &gradients[mid_point..];
        
        // Compress each half differently
        let (sparse_first, _) = self.sparsifier.dgc_compress(first_half, client_id);
        let (quantized_second, norm, _) = self.quantizer.qsgd_compress(second_half);
        
        let mut metadata = HashMap::new();
        metadata.insert("split_point".to_string(), mid_point.to_string());
        
        let hybrid = HybridCompressedGradients {
            method: "layer_wise".to_string(),
            quantized_data: Some(quantized_second),
            sparse_data: Some(sparse_first),
            norm: Some(norm),
            metadata,
        };
        
        let original_size = gradients.len() * 8;
        let compressed_size = quantized_second.len() * 4 + sparse_first.indices.len() * 12 + 8;
        
        let stats = CompressionStats {
            original_size,
            compressed_size,
            compression_ratio: original_size as f64 / compressed_size as f64,
            compression_time: 0.0,
            decompression_time: 0.0,
            accuracy_loss: 0.0,
        };
        
        (hybrid, stats)
    }

    fn compute_natural_sparsity(&self, gradients: &[f64]) -> f64 {
        let threshold = gradients.iter().map(|&x| x.abs()).sum::<f64>() / gradients.len() as f64 * 0.1;
        let sparse_count = gradients.iter().filter(|&&x| x.abs() < threshold).count();
        sparse_count as f64 / gradients.len() as f64
    }

    fn compute_dynamic_range(&self, gradients: &[f64]) -> f64 {
        let max_abs = gradients.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let min_abs = gradients.iter()
            .map(|&x| x.abs())
            .filter(|&x| x > 0.0)
            .fold(f64::INFINITY, f64::min);
        
        if min_abs == f64::INFINITY || min_abs == 0.0 {
            1.0
        } else {
            max_abs / min_abs
        }
    }
}

// Supporting data structures
#[derive(Clone, Debug)]
pub struct SparseGradients {
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct HybridCompressedGradients {
    pub method: String,
    pub quantized_data: Option<Vec<u32>>,
    pub sparse_data: Option<SparseGradients>,
    pub norm: Option<f64>,
    pub metadata: HashMap<String, String>,
}

#[derive(Clone, Debug)]
struct GradientStats {
    count: usize,
    sum: f64,
    sum_squares: f64,
    min: f64,
    max: f64,
    mean: f64,
    variance: f64,
    std_dev: f64,
}

impl GradientStats {
    fn new() -> Self {
        GradientStats {
            count: 0,
            sum: 0.0,
            sum_squares: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squares += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn finalize(&mut self) {
        if self.count > 0 {
            self.mean = self.sum / self.count as f64;
            self.variance = (self.sum_squares / self.count as f64) - (self.mean * self.mean);
            self.std_dev = self.variance.sqrt();
        }
    }
}

// Compression benchmarking
pub fn benchmark_compression_methods(gradients: &[f64]) -> Vec<(String, CompressionStats)> {
    let mut results = Vec::new();
    
    // Test different quantization levels
    for bits in [1, 2, 4, 8, 16] {
        let mut quantizer = QuantizationCompressor::new(bits, false);
        let (_, stats) = quantizer.adaptive_quantize(gradients, 0.01);
        results.push((format!("Quantization-{}bit", bits), stats));
    }
    
    // Test different sparsification ratios
    for sparsity in [0.5, 0.7, 0.9, 0.95, 0.99] {
        let mut sparsifier = SparsificationCompressor::new(sparsity, SparsificationMethod::TopK);
        let (_, stats) = sparsifier.dgc_compress(gradients, "test_client");
        results.push((format!("Sparsification-{:.0}%", sparsity * 100.0), stats));
    }
    
    // Test hybrid methods
    let mut hybrid = HybridCompressor::new(8, 0.9, CompressionStrategy::Adaptive);
    let (_, stats) = hybrid.compress(gradients, "test_client");
    results.push(("Hybrid-Adaptive".to_string(), stats));
    
    results
}

use rand::seq::SliceRandom;