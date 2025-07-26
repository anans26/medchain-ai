use candid::CandidType;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use sha2::{Digest, Sha256};

pub mod rare_diseases;
pub mod validation;
pub mod privacy;

// Core patient data structure
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Patient {
    pub id: String,
    pub identifier: Vec<Identifier>,
    pub name: Vec<HumanName>,
    pub gender: Option<Gender>,
    pub birth_date: Option<String>,
    pub address: Vec<Address>,
    pub contact: Vec<ContactPoint>,
    pub deceased: Option<bool>,
    pub marital_status: Option<CodeableConcept>,
    pub communication: Vec<Communication>,
    pub general_practitioner: Vec<Reference>,
    pub managing_organization: Option<Reference>,
    pub link: Vec<PatientLink>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Identifier {
    pub use_type: Option<String>,
    pub type_code: Option<CodeableConcept>,
    pub system: Option<String>,
    pub value: String,
    pub period: Option<Period>,
    pub assigner: Option<Reference>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct HumanName {
    pub use_type: Option<String>,
    pub text: Option<String>,
    pub family: Option<String>,
    pub given: Vec<String>,
    pub prefix: Vec<String>,
    pub suffix: Vec<String>,
    pub period: Option<Period>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum Gender {
    Male,
    Female,
    Other,
    Unknown,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Address {
    pub use_type: Option<String>,
    pub address_type: Option<String>,
    pub text: Option<String>,
    pub line: Vec<String>,
    pub city: Option<String>,
    pub district: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
    pub period: Option<Period>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct ContactPoint {
    pub system: Option<String>,
    pub value: Option<String>,
    pub use_type: Option<String>,
    pub rank: Option<u32>,
    pub period: Option<Period>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Communication {
    pub language: CodeableConcept,
    pub preferred: Option<bool>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct PatientLink {
    pub other: Reference,
    pub link_type: String,
}

// Medical observation structure
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Observation {
    pub id: String,
    pub identifier: Vec<Identifier>,
    pub status: ObservationStatus,
    pub category: Vec<CodeableConcept>,
    pub code: CodeableConcept,
    pub subject: Reference,
    pub encounter: Option<Reference>,
    pub effective_datetime: Option<String>,
    pub issued: Option<String>,
    pub performer: Vec<Reference>,
    pub value: Option<ObservationValue>,
    pub data_absent_reason: Option<CodeableConcept>,
    pub interpretation: Vec<CodeableConcept>,
    pub note: Vec<Annotation>,
    pub body_site: Option<CodeableConcept>,
    pub method: Option<CodeableConcept>,
    pub specimen: Option<Reference>,
    pub device: Option<Reference>,
    pub reference_range: Vec<ReferenceRange>,
    pub has_member: Vec<Reference>,
    pub derived_from: Vec<Reference>,
    pub component: Vec<ObservationComponent>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum ObservationStatus {
    Registered,
    Preliminary,
    Final,
    Amended,
    Corrected,
    Cancelled,
    EnteredInError,
    Unknown,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum ObservationValue {
    Quantity(Quantity),
    CodeableConcept(CodeableConcept),
    String(String),
    Boolean(bool),
    Integer(i32),
    Range(Range),
    Ratio(Ratio),
    SampledData(SampledData),
    Time(String),
    DateTime(String),
    Period(Period),
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct ObservationComponent {
    pub code: CodeableConcept,
    pub value: Option<ObservationValue>,
    pub data_absent_reason: Option<CodeableConcept>,
    pub interpretation: Vec<CodeableConcept>,
    pub reference_range: Vec<ReferenceRange>,
}

// Diagnostic report structure
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct DiagnosticReport {
    pub id: String,
    pub identifier: Vec<Identifier>,
    pub based_on: Vec<Reference>,
    pub status: DiagnosticReportStatus,
    pub category: Vec<CodeableConcept>,
    pub code: CodeableConcept,
    pub subject: Reference,
    pub encounter: Option<Reference>,
    pub effective_datetime: Option<String>,
    pub issued: Option<String>,
    pub performer: Vec<Reference>,
    pub results_interpreter: Vec<Reference>,
    pub specimen: Vec<Reference>,
    pub result: Vec<Reference>,
    pub imaging_study: Vec<Reference>,
    pub media: Vec<DiagnosticReportMedia>,
    pub conclusion: Option<String>,
    pub conclusion_code: Vec<CodeableConcept>,
    pub presented_form: Vec<Attachment>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum DiagnosticReportStatus {
    Registered,
    Partial,
    Preliminary,
    Final,
    Amended,
    Corrected,
    Appended,
    Cancelled,
    EnteredInError,
    Unknown,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct DiagnosticReportMedia {
    pub comment: Option<String>,
    pub link: Reference,
}

// Condition (diagnosis) structure
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Condition {
    pub id: String,
    pub identifier: Vec<Identifier>,
    pub clinical_status: Option<CodeableConcept>,
    pub verification_status: Option<CodeableConcept>,
    pub category: Vec<CodeableConcept>,
    pub severity: Option<CodeableConcept>,
    pub code: Option<CodeableConcept>,
    pub body_site: Vec<CodeableConcept>,
    pub subject: Reference,
    pub encounter: Option<Reference>,
    pub onset: Option<ConditionOnset>,
    pub abatement: Option<ConditionAbatement>,
    pub recorded_date: Option<String>,
    pub recorder: Option<Reference>,
    pub asserter: Option<Reference>,
    pub stage: Vec<ConditionStage>,
    pub evidence: Vec<ConditionEvidence>,
    pub note: Vec<Annotation>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum ConditionOnset {
    DateTime(String),
    Age(Quantity),
    Period(Period),
    Range(Range),
    String(String),
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum ConditionAbatement {
    DateTime(String),
    Age(Quantity),
    Period(Period),
    Range(Range),
    String(String),
    Boolean(bool),
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct ConditionStage {
    pub summary: Option<CodeableConcept>,
    pub assessment: Vec<Reference>,
    pub stage_type: Option<CodeableConcept>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct ConditionEvidence {
    pub code: Vec<CodeableConcept>,
    pub detail: Vec<Reference>,
}

// Common FHIR data types
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct CodeableConcept {
    pub coding: Vec<Coding>,
    pub text: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Coding {
    pub system: Option<String>,
    pub version: Option<String>,
    pub code: Option<String>,
    pub display: Option<String>,
    pub user_selected: Option<bool>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Reference {
    pub reference: Option<String>,
    pub reference_type: Option<String>,
    pub identifier: Option<Identifier>,
    pub display: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Period {
    pub start: Option<String>,
    pub end: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Quantity {
    pub value: Option<f64>,
    pub comparator: Option<String>,
    pub unit: Option<String>,
    pub system: Option<String>,
    pub code: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Range {
    pub low: Option<Quantity>,
    pub high: Option<Quantity>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Ratio {
    pub numerator: Option<Quantity>,
    pub denominator: Option<Quantity>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct SampledData {
    pub origin: Quantity,
    pub period: f64,
    pub factor: Option<f64>,
    pub lower_limit: Option<f64>,
    pub upper_limit: Option<f64>,
    pub dimensions: u32,
    pub data: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Annotation {
    pub author: Option<AnnotationAuthor>,
    pub time: Option<String>,
    pub text: String,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum AnnotationAuthor {
    Reference(Reference),
    String(String),
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct ReferenceRange {
    pub low: Option<Quantity>,
    pub high: Option<Quantity>,
    pub range_type: Option<CodeableConcept>,
    pub applies_to: Vec<CodeableConcept>,
    pub age: Option<Range>,
    pub text: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Attachment {
    pub content_type: Option<String>,
    pub language: Option<String>,
    pub data: Option<String>,
    pub url: Option<String>,
    pub size: Option<u64>,
    pub hash: Option<String>,
    pub title: Option<String>,
    pub creation: Option<String>,
}

// Medical data processing and validation
impl Patient {
    pub fn new(id: String) -> Self {
        Patient {
            id,
            identifier: Vec::new(),
            name: Vec::new(),
            gender: None,
            birth_date: None,
            address: Vec::new(),
            contact: Vec::new(),
            deceased: None,
            marital_status: None,
            communication: Vec::new(),
            general_practitioner: Vec::new(),
            managing_organization: None,
            link: Vec::new(),
        }
    }

    pub fn add_identifier(&mut self, identifier: Identifier) {
        self.identifier.push(identifier);
    }

    pub fn add_name(&mut self, name: HumanName) {
        self.name.push(name);
    }

    pub fn set_gender(&mut self, gender: Gender) {
        self.gender = Some(gender);
    }

    pub fn set_birth_date(&mut self, birth_date: String) {
        self.birth_date = Some(birth_date);
    }

    pub fn add_address(&mut self, address: Address) {
        self.address.push(address);
    }

    pub fn add_contact(&mut self, contact: ContactPoint) {
        self.contact.push(contact);
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("Patient ID is required".to_string());
        }

        if self.name.is_empty() {
            return Err("Patient must have at least one name".to_string());
        }

        // Validate birth date format if present
        if let Some(ref birth_date) = self.birth_date {
            if !validation::is_valid_date(birth_date) {
                return Err("Invalid birth date format".to_string());
            }
        }

        Ok(())
    }

    pub fn anonymize(&mut self) -> String {
        // Generate a hash-based anonymous ID
        let mut hasher = Sha256::new();
        hasher.update(&self.id);
        let anonymous_id = format!("{:x}", hasher.finalize())[..16].to_string();

        // Clear identifying information
        self.id = anonymous_id.clone();
        self.identifier.clear();
        
        // Anonymize names
        for name in &mut self.name {
            name.family = Some("ANONYMOUS".to_string());
            name.given = vec!["PATIENT".to_string()];
            name.text = Some("ANONYMOUS PATIENT".to_string());
        }

        // Clear addresses
        for address in &mut self.address {
            address.line.clear();
            address.city = Some("ANONYMOUS".to_string());
            address.postal_code = None;
        }

        // Clear contact information
        self.contact.clear();

        anonymous_id
    }
}

impl Observation {
    pub fn new(id: String, code: CodeableConcept, subject: Reference) -> Self {
        Observation {
            id,
            identifier: Vec::new(),
            status: ObservationStatus::Final,
            category: Vec::new(),
            code,
            subject,
            encounter: None,
            effective_datetime: None,
            issued: None,
            performer: Vec::new(),
            value: None,
            data_absent_reason: None,
            interpretation: Vec::new(),
            note: Vec::new(),
            body_site: None,
            method: None,
            specimen: None,
            device: None,
            reference_range: Vec::new(),
            has_member: Vec::new(),
            derived_from: Vec::new(),
            component: Vec::new(),
        }
    }

    pub fn set_value(&mut self, value: ObservationValue) {
        self.value = Some(value);
    }

    pub fn set_status(&mut self, status: ObservationStatus) {
        self.status = status;
    }

    pub fn add_category(&mut self, category: CodeableConcept) {
        self.category.push(category);
    }

    pub fn add_interpretation(&mut self, interpretation: CodeableConcept) {
        self.interpretation.push(interpretation);
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("Observation ID is required".to_string());
        }

        if self.code.coding.is_empty() && self.code.text.is_none() {
            return Err("Observation code is required".to_string());
        }

        if self.subject.reference.is_none() && self.subject.identifier.is_none() {
            return Err("Observation subject is required".to_string());
        }

        Ok(())
    }
}

impl Condition {
    pub fn new(id: String, subject: Reference) -> Self {
        Condition {
            id,
            identifier: Vec::new(),
            clinical_status: None,
            verification_status: None,
            category: Vec::new(),
            severity: None,
            code: None,
            body_site: Vec::new(),
            subject,
            encounter: None,
            onset: None,
            abatement: None,
            recorded_date: None,
            recorder: None,
            asserter: None,
            stage: Vec::new(),
            evidence: Vec::new(),
            note: Vec::new(),
        }
    }

    pub fn set_code(&mut self, code: CodeableConcept) {
        self.code = Some(code);
    }

    pub fn set_clinical_status(&mut self, status: CodeableConcept) {
        self.clinical_status = Some(status);
    }

    pub fn set_verification_status(&mut self, status: CodeableConcept) {
        self.verification_status = Some(status);
    }

    pub fn add_category(&mut self, category: CodeableConcept) {
        self.category.push(category);
    }

    pub fn set_severity(&mut self, severity: CodeableConcept) {
        self.severity = Some(severity);
    }

    pub fn add_body_site(&mut self, body_site: CodeableConcept) {
        self.body_site.push(body_site);
    }

    pub fn set_onset(&mut self, onset: ConditionOnset) {
        self.onset = Some(onset);
    }

    pub fn add_evidence(&mut self, evidence: ConditionEvidence) {
        self.evidence.push(evidence);
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("Condition ID is required".to_string());
        }

        if self.subject.reference.is_none() && self.subject.identifier.is_none() {
            return Err("Condition subject is required".to_string());
        }

        Ok(())
    }
}

// Helper functions for creating common medical concepts
pub fn create_coding(system: &str, code: &str, display: &str) -> Coding {
    Coding {
        system: Some(system.to_string()),
        version: None,
        code: Some(code.to_string()),
        display: Some(display.to_string()),
        user_selected: None,
    }
}

pub fn create_codeable_concept(coding: Coding, text: Option<&str>) -> CodeableConcept {
    CodeableConcept {
        coding: vec![coding],
        text: text.map(|t| t.to_string()),
    }
}

pub fn create_reference(reference: &str, display: Option<&str>) -> Reference {
    Reference {
        reference: Some(reference.to_string()),
        reference_type: None,
        identifier: None,
        display: display.map(|d| d.to_string()),
    }
}

pub fn create_quantity(value: f64, unit: &str, system: Option<&str>, code: Option<&str>) -> Quantity {
    Quantity {
        value: Some(value),
        comparator: None,
        unit: Some(unit.to_string()),
        system: system.map(|s| s.to_string()),
        code: code.map(|c| c.to_string()),
    }
}

// Medical data aggregation for AI training
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct MedicalDataset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub patients: Vec<Patient>,
    pub observations: Vec<Observation>,
    pub conditions: Vec<Condition>,
    pub diagnostic_reports: Vec<DiagnosticReport>,
    pub created_at: String,
    pub updated_at: String,
    pub version: String,
    pub metadata: HashMap<String, String>,
}

impl MedicalDataset {
    pub fn new(id: String, name: String, description: String) -> Self {
        let now = Utc::now().to_rfc3339();
        MedicalDataset {
            id,
            name,
            description,
            patients: Vec::new(),
            observations: Vec::new(),
            conditions: Vec::new(),
            diagnostic_reports: Vec::new(),
            created_at: now.clone(),
            updated_at: now,
            version: "1.0.0".to_string(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_patient(&mut self, patient: Patient) -> Result<(), String> {
        patient.validate()?;
        self.patients.push(patient);
        self.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    pub fn add_observation(&mut self, observation: Observation) -> Result<(), String> {
        observation.validate()?;
        self.observations.push(observation);
        self.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    pub fn add_condition(&mut self, condition: Condition) -> Result<(), String> {
        condition.validate()?;
        self.conditions.push(condition);
        self.updated_at = Utc::now().to_rfc3339();
        Ok(())
    }

    pub fn add_diagnostic_report(&mut self, report: DiagnosticReport) {
        self.diagnostic_reports.push(report);
        self.updated_at = Utc::now().to_rfc3339();
    }

    pub fn get_patient_count(&self) -> usize {
        self.patients.len()
    }

    pub fn get_observation_count(&self) -> usize {
        self.observations.len()
    }

    pub fn get_condition_count(&self) -> usize {
        self.conditions.len()
    }

    pub fn anonymize_dataset(&mut self) -> HashMap<String, String> {
        let mut id_mapping = HashMap::new();

        // Anonymize patients
        for patient in &mut self.patients {
            let original_id = patient.id.clone();
            let anonymous_id = patient.anonymize();
            id_mapping.insert(original_id, anonymous_id);
        }

        // Update references in observations
        for observation in &mut self.observations {
            if let Some(ref mut subject_ref) = observation.subject.reference {
                if let Some(anonymous_id) = id_mapping.get(subject_ref) {
                    *subject_ref = format!("Patient/{}", anonymous_id);
                }
            }
        }

        // Update references in conditions
        for condition in &mut self.conditions {
            if let Some(ref mut subject_ref) = condition.subject.reference {
                if let Some(anonymous_id) = id_mapping.get(subject_ref) {
                    *subject_ref = format!("Patient/{}", anonymous_id);
                }
            }
        }

        self.updated_at = Utc::now().to_rfc3339();
        id_mapping
    }

    pub fn validate_dataset(&self) -> Result<(), String> {
        // Validate all patients
        for patient in &self.patients {
            patient.validate()?;
        }

        // Validate all observations
        for observation in &self.observations {
            observation.validate()?;
        }

        // Validate all conditions
        for condition in &self.conditions {
            condition.validate()?;
        }

        Ok(())
    }

    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        stats.insert("patient_count".to_string(), serde_json::Value::Number(self.patients.len().into()));
        stats.insert("observation_count".to_string(), serde_json::Value::Number(self.observations.len().into()));
        stats.insert("condition_count".to_string(), serde_json::Value::Number(self.conditions.len().into()));
        stats.insert("diagnostic_report_count".to_string(), serde_json::Value::Number(self.diagnostic_reports.len().into()));
        
        // Gender distribution
        let mut gender_counts = HashMap::new();
        for patient in &self.patients {
            let gender_key = match &patient.gender {
                Some(Gender::Male) => "male",
                Some(Gender::Female) => "female",
                Some(Gender::Other) => "other",
                Some(Gender::Unknown) => "unknown",
                None => "not_specified",
            };
            *gender_counts.entry(gender_key).or_insert(0) += 1;
        }
        stats.insert("gender_distribution".to_string(), serde_json::to_value(gender_counts).unwrap());

        // Age distribution (if birth dates are available)
        let mut age_groups = HashMap::new();
        let current_year = Utc::now().year();
        for patient in &self.patients {
            if let Some(ref birth_date) = patient.birth_date {
                if let Ok(birth_year) = birth_date[..4].parse::<i32>() {
                    let age = current_year - birth_year;
                    let age_group = match age {
                        0..=17 => "0-17",
                        18..=34 => "18-34",
                        35..=54 => "35-54",
                        55..=74 => "55-74",
                        _ => "75+",
                    };
                    *age_groups.entry(age_group).or_insert(0) += 1;
                }
            }
        }
        stats.insert("age_distribution".to_string(), serde_json::to_value(age_groups).unwrap());

        stats
    }
}