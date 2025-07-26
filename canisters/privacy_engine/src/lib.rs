use ic_cdk::api::management_canister::main::raw_rand;
use ic_cdk_macros::{init, post_upgrade, pre_upgrade, query, update};
use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use ic_stable_structures::{DefaultMemoryImpl, StableBTreeMap, Storable};
use std::borrow::Cow;
use std::cell::RefCell;
use candid::{CandidType, Decode, Encode, Principal};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use differential_privacy::DifferentialPrivacy;

type Memory = VirtualMemory<DefaultMemoryImpl>;

// Privacy budget tracking for hospitals
#[derive(CandidType, Serialize, Deserialize, Clone)]
pub struct PrivacyBudget {
    pub hospital_id: Principal,
    pub epsilon_used: f64,
    pub epsilon_total: f64,
    pub delta_used: f64,
    pub delta_total: f64,
    pub last_updated: u64,
    pub queries_count: u64,
}

impl Storable for PrivacyBudget {
    fn to_bytes(&self) -> Cow<[u8]> {
        Cow::Owned(Encode!(self).unwrap())
    }

    fn from_bytes(bytes: Cow<[u8]>) -> Self {
        Decode!(bytes.as_ref(), Self).unwrap()
    }
}

// Privacy audit log entry
#[derive(CandidType, Serialize, Deserialize, Clone)]
pub struct PrivacyAuditEntry {
    pub id: u64,
    pub hospital_id: Principal,
    pub operation_type: String,
    pub epsilon_consumed: f64,
    pub delta_consumed: f64,
    pub timestamp: u64,
    pub data_hash: String,
    pub compliance_status: ComplianceStatus,
}

#[derive(CandidType, Serialize, Deserialize, Clone)]
pub enum ComplianceStatus {
    Compliant,
    Warning,
    Violation,
}

impl Storable for PrivacyAuditEntry {
    fn to_bytes(&self) -> Cow<[u8]> {
        Cow::Owned(Encode!(self).unwrap())
    }

    fn from_bytes(bytes: Cow<[u8]>) -> Self {
        Decode!(bytes.as_ref(), Self).unwrap()
    }
}

// Cross-canister privacy coordination
#[derive(CandidType, Serialize, Deserialize, Clone)]
pub struct PrivacyCoordination {
    pub session_id: String,
    pub participating_hospitals: Vec<Principal>,
    pub total_epsilon_budget: f64,
    pub allocated_budgets: Vec<(Principal, f64)>,
    pub status: CoordinationStatus,
    pub created_at: u64,
}

#[derive(CandidType, Serialize, Deserialize, Clone)]
pub enum CoordinationStatus {
    Pending,
    Active,
    Completed,
    Failed,
}

impl Storable for PrivacyCoordination {
    fn to_bytes(&self) -> Cow<[u8]> {
        Cow::Owned(Encode!(self).unwrap())
    }

    fn from_bytes(bytes: Cow<[u8]>) -> Self {
        Decode!(bytes.as_ref(), Self).unwrap()
    }
}

// Global state management
thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    static PRIVACY_BUDGETS: RefCell<StableBTreeMap<Principal, PrivacyBudget, Memory>> = RefCell::new(
        StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0))),
        )
    );

    static AUDIT_LOG: RefCell<StableBTreeMap<u64, PrivacyAuditEntry, Memory>> = RefCell::new(
        StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1))),
        )
    );

    static PRIVACY_COORDINATIONS: RefCell<StableBTreeMap<String, PrivacyCoordination, Memory>> = RefCell::new(
        StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(2))),
        )
    );

    static DIFFERENTIAL_PRIVACY: RefCell<DifferentialPrivacy> = RefCell::new(DifferentialPrivacy::new());
    static AUDIT_COUNTER: RefCell<u64> = RefCell::new(0);
}

#[init]
fn init() {
    ic_cdk::println!("Privacy Engine initialized");
}

#[pre_upgrade]
fn pre_upgrade() {
    // Stable memory automatically persists data
}

#[post_upgrade]
fn post_upgrade() {
    ic_cdk::println!("Privacy Engine upgraded");
}

// Hospital registration and privacy budget allocation
#[update]
async fn register_hospital(hospital_id: Principal, epsilon_total: f64, delta_total: f64) -> Result<String, String> {
    let caller = ic_cdk::caller();
    
    // In production, add proper authorization checks
    if caller == Principal::anonymous() {
        return Err("Anonymous caller not allowed".to_string());
    }

    let privacy_budget = PrivacyBudget {
        hospital_id,
        epsilon_used: 0.0,
        epsilon_total,
        delta_used: 0.0,
        delta_total,
        last_updated: ic_cdk::api::time(),
        queries_count: 0,
    };

    PRIVACY_BUDGETS.with(|budgets| {
        budgets.borrow_mut().insert(hospital_id, privacy_budget);
    });

    // Log the registration
    log_privacy_audit(
        hospital_id,
        "hospital_registration".to_string(),
        0.0,
        0.0,
        "".to_string(),
        ComplianceStatus::Compliant,
    ).await;

    Ok(format!("Hospital {} registered with privacy budget ε={}, δ={}", hospital_id, epsilon_total, delta_total))
}

// Check if a privacy operation is allowed
#[query]
fn check_privacy_budget(hospital_id: Principal, epsilon_required: f64, delta_required: f64) -> Result<bool, String> {
    PRIVACY_BUDGETS.with(|budgets| {
        match budgets.borrow().get(&hospital_id) {
            Some(budget) => {
                let epsilon_available = budget.epsilon_total - budget.epsilon_used;
                let delta_available = budget.delta_total - budget.delta_used;
                
                Ok(epsilon_available >= epsilon_required && delta_available >= delta_required)
            }
            None => Err("Hospital not registered".to_string())
        }
    })
}

// Consume privacy budget for an operation
#[update]
async fn consume_privacy_budget(
    hospital_id: Principal,
    epsilon_consumed: f64,
    delta_consumed: f64,
    operation_type: String,
    data_hash: String,
) -> Result<String, String> {
    let caller = ic_cdk::caller();
    
    // Verify the caller is authorized (in production, implement proper auth)
    if caller == Principal::anonymous() {
        return Err("Anonymous caller not allowed".to_string());
    }

    PRIVACY_BUDGETS.with(|budgets| {
        let mut budgets_map = budgets.borrow_mut();
        match budgets_map.get(&hospital_id) {
            Some(mut budget) => {
                let epsilon_available = budget.epsilon_total - budget.epsilon_used;
                let delta_available = budget.delta_total - budget.delta_used;
                
                if epsilon_available < epsilon_consumed || delta_available < delta_consumed {
                    return Err("Insufficient privacy budget".to_string());
                }

                // Update budget
                budget.epsilon_used += epsilon_consumed;
                budget.delta_used += delta_consumed;
                budget.last_updated = ic_cdk::api::time();
                budget.queries_count += 1;

                budgets_map.insert(hospital_id, budget);

                // Determine compliance status
                let epsilon_usage_ratio = budget.epsilon_used / budget.epsilon_total;
                let compliance_status = if epsilon_usage_ratio > 0.9 {
                    ComplianceStatus::Warning
                } else if epsilon_usage_ratio > 1.0 {
                    ComplianceStatus::Violation
                } else {
                    ComplianceStatus::Compliant
                };

                // Log the operation
                ic_cdk::spawn(log_privacy_audit(
                    hospital_id,
                    operation_type,
                    epsilon_consumed,
                    delta_consumed,
                    data_hash,
                    compliance_status,
                ));

                Ok(format!("Privacy budget consumed: ε={}, δ={}", epsilon_consumed, delta_consumed))
            }
            None => Err("Hospital not registered".to_string())
        }
    })
}

// Get privacy budget status for a hospital
#[query]
fn get_privacy_budget(hospital_id: Principal) -> Result<PrivacyBudget, String> {
    PRIVACY_BUDGETS.with(|budgets| {
        match budgets.borrow().get(&hospital_id) {
            Some(budget) => Ok(budget),
            None => Err("Hospital not registered".to_string())
        }
    })
}

// Coordinate privacy across multiple hospitals for federated learning
#[update]
async fn coordinate_federated_privacy(
    session_id: String,
    participating_hospitals: Vec<Principal>,
    total_epsilon_budget: f64,
) -> Result<String, String> {
    let caller = ic_cdk::caller();
    
    if caller == Principal::anonymous() {
        return Err("Anonymous caller not allowed".to_string());
    }

    // Allocate budget equally among hospitals
    let epsilon_per_hospital = total_epsilon_budget / participating_hospitals.len() as f64;
    let mut allocated_budgets = Vec::new();

    // Check if all hospitals have sufficient budget
    for hospital_id in &participating_hospitals {
        match check_privacy_budget(*hospital_id, epsilon_per_hospital, 1e-5) {
            Ok(true) => {
                allocated_budgets.push((*hospital_id, epsilon_per_hospital));
            }
            Ok(false) => {
                return Err(format!("Hospital {} has insufficient privacy budget", hospital_id));
            }
            Err(e) => return Err(e),
        }
    }

    let coordination = PrivacyCoordination {
        session_id: session_id.clone(),
        participating_hospitals,
        total_epsilon_budget,
        allocated_budgets,
        status: CoordinationStatus::Active,
        created_at: ic_cdk::api::time(),
    };

    PRIVACY_COORDINATIONS.with(|coords| {
        coords.borrow_mut().insert(session_id.clone(), coordination);
    });

    Ok(format!("Privacy coordination established for session {}", session_id))
}

// Add noise to gradients using differential privacy
#[update]
async fn add_privacy_noise(
    hospital_id: Principal,
    gradients: Vec<f64>,
    epsilon: f64,
    delta: f64,
    sensitivity: f64,
) -> Result<Vec<f64>, String> {
    let caller = ic_cdk::caller();
    
    if caller == Principal::anonymous() {
        return Err("Anonymous caller not allowed".to_string());
    }

    // Check privacy budget
    match check_privacy_budget(hospital_id, epsilon, delta) {
        Ok(true) => {},
        Ok(false) => return Err("Insufficient privacy budget".to_string()),
        Err(e) => return Err(e),
    }

    // Add differential privacy noise
    let noisy_gradients = DIFFERENTIAL_PRIVACY.with(|dp| {
        let dp_instance = dp.borrow();
        gradients.iter().map(|&gradient| {
            gradient + dp_instance.add_gaussian_noise(sensitivity, epsilon, delta)
        }).collect()
    });

    // Consume privacy budget
    let data_hash = compute_hash(&gradients);
    consume_privacy_budget(
        hospital_id,
        epsilon,
        delta,
        "gradient_noise_addition".to_string(),
        data_hash,
    ).await?;

    Ok(noisy_gradients)
}

// Generate privacy audit report
#[query]
fn get_privacy_audit_report(hospital_id: Option<Principal>, limit: Option<u64>) -> Vec<PrivacyAuditEntry> {
    let limit = limit.unwrap_or(100);
    
    AUDIT_LOG.with(|log| {
        let log_map = log.borrow();
        let mut entries: Vec<PrivacyAuditEntry> = log_map
            .iter()
            .map(|(_, entry)| entry)
            .filter(|entry| {
                hospital_id.map_or(true, |id| entry.hospital_id == id)
            })
            .collect();
        
        // Sort by timestamp (most recent first)
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(limit as usize);
        entries
    })
}

// Check overall system compliance
#[query]
fn check_system_compliance() -> Result<String, String> {
    let mut total_hospitals = 0;
    let mut compliant_hospitals = 0;
    let mut warning_hospitals = 0;
    let mut violation_hospitals = 0;

    PRIVACY_BUDGETS.with(|budgets| {
        for (_, budget) in budgets.borrow().iter() {
            total_hospitals += 1;
            let usage_ratio = budget.epsilon_used / budget.epsilon_total;
            
            if usage_ratio > 1.0 {
                violation_hospitals += 1;
            } else if usage_ratio > 0.9 {
                warning_hospitals += 1;
            } else {
                compliant_hospitals += 1;
            }
        }
    });

    let compliance_report = format!(
        "System Compliance Report:\nTotal Hospitals: {}\nCompliant: {}\nWarning: {}\nViolations: {}",
        total_hospitals, compliant_hospitals, warning_hospitals, violation_hospitals
    );

    Ok(compliance_report)
}

// Helper function to log privacy audit entries
async fn log_privacy_audit(
    hospital_id: Principal,
    operation_type: String,
    epsilon_consumed: f64,
    delta_consumed: f64,
    data_hash: String,
    compliance_status: ComplianceStatus,
) {
    let audit_id = AUDIT_COUNTER.with(|counter| {
        let mut c = counter.borrow_mut();
        *c += 1;
        *c
    });

    let audit_entry = PrivacyAuditEntry {
        id: audit_id,
        hospital_id,
        operation_type,
        epsilon_consumed,
        delta_consumed,
        timestamp: ic_cdk::api::time(),
        data_hash,
        compliance_status,
    };

    AUDIT_LOG.with(|log| {
        log.borrow_mut().insert(audit_id, audit_entry);
    });
}

// Helper function to compute hash of data
fn compute_hash(data: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for &value in data {
        hasher.update(value.to_be_bytes());
    }
    format!("{:x}", hasher.finalize())
}

// Reset privacy budget (admin function - use with caution)
#[update]
async fn reset_privacy_budget(hospital_id: Principal) -> Result<String, String> {
    let caller = ic_cdk::caller();
    
    // In production, implement proper admin authorization
    if caller == Principal::anonymous() {
        return Err("Anonymous caller not allowed".to_string());
    }

    PRIVACY_BUDGETS.with(|budgets| {
        let mut budgets_map = budgets.borrow_mut();
        match budgets_map.get(&hospital_id) {
            Some(mut budget) => {
                budget.epsilon_used = 0.0;
                budget.delta_used = 0.0;
                budget.last_updated = ic_cdk::api::time();
                budget.queries_count = 0;
                
                budgets_map.insert(hospital_id, budget);

                // Log the reset
                ic_cdk::spawn(log_privacy_audit(
                    hospital_id,
                    "budget_reset".to_string(),
                    0.0,
                    0.0,
                    "".to_string(),
                    ComplianceStatus::Compliant,
                ));

                Ok(format!("Privacy budget reset for hospital {}", hospital_id))
            }
            None => Err("Hospital not registered".to_string())
        }
    })
}

// Export Candid interface
ic_cdk::export_candid!();