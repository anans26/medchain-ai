use crate::*;
use serde::{Deserialize, Serialize};
use candid::CandidType;
use std::collections::HashMap;

// Rare disease classification system
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct RareDisease {
    pub orpha_code: String,
    pub name: String,
    pub synonyms: Vec<String>,
    pub definition: String,
    pub prevalence: Prevalence,
    pub inheritance_pattern: Vec<InheritancePattern>,
    pub age_of_onset: Vec<AgeOfOnset>,
    pub clinical_features: Vec<ClinicalFeature>,
    pub diagnostic_criteria: Vec<DiagnosticCriterion>,
    pub differential_diagnosis: Vec<String>,
    pub genes: Vec<Gene>,
    pub phenotypes: Vec<Phenotype>,
    pub icd10_codes: Vec<String>,
    pub icd11_codes: Vec<String>,
    pub omim_codes: Vec<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Prevalence {
    pub point_prevalence: Option<f64>,
    pub birth_prevalence: Option<f64>,
    pub lifetime_prevalence: Option<f64>,
    pub prevalence_class: PrevalenceClass,
    pub geographic_distribution: Vec<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum PrevalenceClass {
    VeryRare,      // <1/1,000,000
    Rare,          // 1-9/1,000,000
    ModeratelyRare, // 1-9/100,000
    Unknown,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum InheritancePattern {
    AutosomalDominant,
    AutosomalRecessive,
    XLinkedDominant,
    XLinkedRecessive,
    YLinked,
    Mitochondrial,
    Multifactorial,
    Somatic,
    Unknown,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum AgeOfOnset {
    Antenatal,
    Neonatal,
    Infancy,
    Childhood,
    Adolescent,
    Adult,
    Elderly,
    AllAges,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct ClinicalFeature {
    pub hpo_id: String,
    pub name: String,
    pub frequency: Frequency,
    pub severity: Option<Severity>,
    pub body_system: BodySystem,
    pub description: String,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum Frequency {
    Obligate,      // 100%
    VeryFrequent,  // 80-99%
    Frequent,      // 30-79%
    Occasional,    // 5-29%
    VeryRare,      // <5%
    Excluded,      // 0%
    Unknown,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum Severity {
    Mild,
    Moderate,
    Severe,
    Profound,
    Variable,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum BodySystem {
    Cardiovascular,
    Respiratory,
    Gastrointestinal,
    Genitourinary,
    Musculoskeletal,
    Neurological,
    Endocrine,
    Hematologic,
    Immunologic,
    Dermatologic,
    Ophthalmologic,
    Otolaryngologic,
    Psychiatric,
    Multiple,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct DiagnosticCriterion {
    pub criterion_type: DiagnosticCriterionType,
    pub description: String,
    pub required: bool,
    pub test_type: Option<TestType>,
    pub normal_range: Option<String>,
    pub pathological_range: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum DiagnosticCriterionType {
    Clinical,
    Laboratory,
    Imaging,
    Genetic,
    Histological,
    Functional,
    Molecular,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum TestType {
    BloodTest,
    UrineTest,
    CsfTest,
    Biopsy,
    Xray,
    Ct,
    Mri,
    Ultrasound,
    Ecg,
    Eeg,
    Emg,
    GeneticSequencing,
    Karyotype,
    Fish,
    Pcr,
    WesternBlot,
    Elisa,
    FlowCytometry,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Gene {
    pub symbol: String,
    pub name: String,
    pub hgnc_id: String,
    pub entrez_id: Option<u32>,
    pub ensembl_id: Option<String>,
    pub chromosome: String,
    pub location: String,
    pub function: String,
    pub disease_mechanism: DiseaseMechanism,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum DiseaseMechanism {
    LossOfFunction,
    GainOfFunction,
    DominantNegative,
    Haploinsufficiency,
    Triplet,
    Chromosomal,
    Unknown,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Phenotype {
    pub hpo_id: String,
    pub name: String,
    pub definition: String,
    pub frequency: Frequency,
    pub onset: Option<AgeOfOnset>,
    pub severity: Option<Severity>,
    pub modifiers: Vec<String>,
}

// Rare disease patient case
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct RareDiseaseCase {
    pub case_id: String,
    pub patient: Patient,
    pub presenting_symptoms: Vec<ClinicalFeature>,
    pub family_history: Vec<FamilyHistoryEntry>,
    pub diagnostic_journey: DiagnosticJourney,
    pub confirmed_diagnosis: Option<RareDisease>,
    pub differential_diagnoses: Vec<DifferentialDiagnosis>,
    pub genetic_testing: Vec<GeneticTest>,
    pub treatment_history: Vec<Treatment>,
    pub outcome: Option<CaseOutcome>,
    pub case_notes: Vec<CaseNote>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct FamilyHistoryEntry {
    pub relationship: String,
    pub affected: bool,
    pub condition: Option<String>,
    pub age_of_onset: Option<u32>,
    pub notes: String,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct DiagnosticJourney {
    pub initial_presentation_date: String,
    pub diagnosis_date: Option<String>,
    pub time_to_diagnosis_days: Option<u32>,
    pub physicians_consulted: u32,
    pub misdiagnoses: Vec<String>,
    pub diagnostic_tests: Vec<DiagnosticTest>,
    pub referrals: Vec<Referral>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct DifferentialDiagnosis {
    pub disease: RareDisease,
    pub probability: f64,
    pub supporting_evidence: Vec<String>,
    pub contradicting_evidence: Vec<String>,
    pub ruled_out: bool,
    pub ruled_out_reason: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct GeneticTest {
    pub test_type: GeneticTestType,
    pub genes_tested: Vec<String>,
    pub results: Vec<GeneticVariant>,
    pub interpretation: String,
    pub date_performed: String,
    pub laboratory: String,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum GeneticTestType {
    SingleGene,
    GenePanel,
    WholeExomeSequencing,
    WholeGenomeSequencing,
    Karyotype,
    Microarray,
    MlpaAnalysis,
    SangerSequencing,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct GeneticVariant {
    pub gene: String,
    pub variant: String,
    pub zygosity: Zygosity,
    pub classification: VariantClassification,
    pub inheritance: Option<InheritancePattern>,
    pub population_frequency: Option<f64>,
    pub pathogenicity_score: Option<f64>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum Zygosity {
    Homozygous,
    Heterozygous,
    Hemizygous,
    Compound,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum VariantClassification {
    Pathogenic,
    LikelyPathogenic,
    VariantOfUncertainSignificance,
    LikelyBenign,
    Benign,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct DiagnosticTest {
    pub test_name: String,
    pub test_type: TestType,
    pub date_performed: String,
    pub results: String,
    pub normal_range: Option<String>,
    pub interpretation: String,
    pub ordering_physician: String,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Referral {
    pub specialty: String,
    pub physician_name: String,
    pub date: String,
    pub reason: String,
    pub outcome: String,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct Treatment {
    pub treatment_type: TreatmentType,
    pub medication: Option<String>,
    pub dosage: Option<String>,
    pub start_date: String,
    pub end_date: Option<String>,
    pub response: TreatmentResponse,
    pub side_effects: Vec<String>,
    pub notes: String,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum TreatmentType {
    Medication,
    Surgery,
    PhysicalTherapy,
    OccupationalTherapy,
    SpeechTherapy,
    DietaryModification,
    GeneTherapy,
    StemCellTherapy,
    Supportive,
    Palliative,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum TreatmentResponse {
    Excellent,
    Good,
    Partial,
    Minimal,
    None,
    Adverse,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct CaseOutcome {
    pub status: CaseStatus,
    pub quality_of_life_score: Option<f64>,
    pub functional_status: String,
    pub prognosis: Prognosis,
    pub follow_up_required: bool,
    pub last_follow_up: Option<String>,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum CaseStatus {
    Diagnosed,
    UnderInvestigation,
    Undiagnosed,
    Deceased,
    LostToFollowUp,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum Prognosis {
    Excellent,
    Good,
    Fair,
    Poor,
    Terminal,
    Unknown,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct CaseNote {
    pub date: String,
    pub author: String,
    pub note_type: NoteType,
    pub content: String,
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum NoteType {
    Clinical,
    Genetic,
    Laboratory,
    Imaging,
    Treatment,
    FollowUp,
    Research,
}

// Rare disease database and utilities
pub struct RareDiseaseDatabase {
    diseases: HashMap<String, RareDisease>,
    cases: HashMap<String, RareDiseaseCase>,
}

impl RareDiseaseDatabase {
    pub fn new() -> Self {
        RareDiseaseDatabase {
            diseases: HashMap::new(),
            cases: HashMap::new(),
        }
    }

    pub fn add_disease(&mut self, disease: RareDisease) {
        self.diseases.insert(disease.orpha_code.clone(), disease);
    }

    pub fn add_case(&mut self, case: RareDiseaseCase) {
        self.cases.insert(case.case_id.clone(), case);
    }

    pub fn get_disease(&self, orpha_code: &str) -> Option<&RareDisease> {
        self.diseases.get(orpha_code)
    }

    pub fn get_case(&self, case_id: &str) -> Option<&RareDiseaseCase> {
        self.cases.get(case_id)
    }

    pub fn search_diseases_by_symptoms(&self, symptoms: &[String]) -> Vec<&RareDisease> {
        self.diseases
            .values()
            .filter(|disease| {
                symptoms.iter().any(|symptom| {
                    disease.clinical_features.iter().any(|feature| {
                        feature.name.to_lowercase().contains(&symptom.to_lowercase())
                            || feature.hpo_id == *symptom
                    })
                })
            })
            .collect()
    }

    pub fn search_diseases_by_gene(&self, gene_symbol: &str) -> Vec<&RareDisease> {
        self.diseases
            .values()
            .filter(|disease| {
                disease.genes.iter().any(|gene| gene.symbol == gene_symbol)
            })
            .collect()
    }

    pub fn get_diagnostic_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        let total_cases = self.cases.len() as f64;
        if total_cases == 0.0 {
            return stats;
        }

        // Calculate average time to diagnosis
        let mut total_time_to_diagnosis = 0.0;
        let mut diagnosed_cases = 0.0;
        
        for case in self.cases.values() {
            if let Some(days) = case.diagnostic_journey.time_to_diagnosis_days {
                total_time_to_diagnosis += days as f64;
                diagnosed_cases += 1.0;
            }
        }

        if diagnosed_cases > 0.0 {
            stats.insert("average_time_to_diagnosis_days".to_string(), total_time_to_diagnosis / diagnosed_cases);
        }

        // Calculate diagnosis rate
        let diagnosed_count = self.cases.values()
            .filter(|case| case.confirmed_diagnosis.is_some())
            .count() as f64;
        
        stats.insert("diagnosis_rate".to_string(), diagnosed_count / total_cases);

        // Calculate average physicians consulted
        let total_physicians: u32 = self.cases.values()
            .map(|case| case.diagnostic_journey.physicians_consulted)
            .sum();
        
        stats.insert("average_physicians_consulted".to_string(), total_physicians as f64 / total_cases);

        stats
    }

    pub fn generate_synthetic_case(&self, disease_orpha_code: &str) -> Option<RareDiseaseCase> {
        let disease = self.get_disease(disease_orpha_code)?;
        
        // Create synthetic patient
        let mut patient = Patient::new(format!("synthetic_{}", uuid::Uuid::new_v4()));
        patient.set_gender(Gender::Unknown);
        
        // Add synthetic name
        let mut name = HumanName {
            use_type: Some("official".to_string()),
            text: Some("Synthetic Patient".to_string()),
            family: Some("Patient".to_string()),
            given: vec!["Synthetic".to_string()],
            prefix: Vec::new(),
            suffix: Vec::new(),
            period: None,
        };
        patient.add_name(name);

        // Create presenting symptoms based on disease features
        let presenting_symptoms: Vec<ClinicalFeature> = disease.clinical_features
            .iter()
            .filter(|feature| matches!(feature.frequency, Frequency::Obligate | Frequency::VeryFrequent | Frequency::Frequent))
            .cloned()
            .collect();

        // Create diagnostic journey
        let diagnostic_journey = DiagnosticJourney {
            initial_presentation_date: "2024-01-01".to_string(),
            diagnosis_date: Some("2024-06-01".to_string()),
            time_to_diagnosis_days: Some(150), // ~5 months
            physicians_consulted: 5,
            misdiagnoses: vec!["Common condition".to_string()],
            diagnostic_tests: Vec::new(),
            referrals: Vec::new(),
        };

        Some(RareDiseaseCase {
            case_id: format!("case_{}", uuid::Uuid::new_v4()),
            patient,
            presenting_symptoms,
            family_history: Vec::new(),
            diagnostic_journey,
            confirmed_diagnosis: Some(disease.clone()),
            differential_diagnoses: Vec::new(),
            genetic_testing: Vec::new(),
            treatment_history: Vec::new(),
            outcome: None,
            case_notes: Vec::new(),
        })
    }
}

// Initialize database with common rare diseases
pub fn initialize_rare_disease_database() -> RareDiseaseDatabase {
    let mut db = RareDiseaseDatabase::new();

    // Add Huntington's Disease
    let huntingtons = RareDisease {
        orpha_code: "ORPHA:399".to_string(),
        name: "Huntington disease".to_string(),
        synonyms: vec!["Huntington's chorea".to_string(), "HD".to_string()],
        definition: "A rare neurodegenerative disorder characterized by progressive motor, cognitive, and psychiatric symptoms".to_string(),
        prevalence: Prevalence {
            point_prevalence: Some(5.7e-5),
            birth_prevalence: None,
            lifetime_prevalence: None,
            prevalence_class: PrevalenceClass::Rare,
            geographic_distribution: vec!["Worldwide".to_string()],
        },
        inheritance_pattern: vec![InheritancePattern::AutosomalDominant],
        age_of_onset: vec![AgeOfOnset::Adult],
        clinical_features: vec![
            ClinicalFeature {
                hpo_id: "HP:0002072".to_string(),
                name: "Chorea".to_string(),
                frequency: Frequency::VeryFrequent,
                severity: Some(Severity::Moderate),
                body_system: BodySystem::Neurological,
                description: "Involuntary, irregular, purposeless movements".to_string(),
            },
            ClinicalFeature {
                hpo_id: "HP:0000726".to_string(),
                name: "Dementia".to_string(),
                frequency: Frequency::Frequent,
                severity: Some(Severity::Severe),
                body_system: BodySystem::Neurological,
                description: "Progressive cognitive decline".to_string(),
            },
        ],
        diagnostic_criteria: vec![
            DiagnosticCriterion {
                criterion_type: DiagnosticCriterionType::Genetic,
                description: "CAG repeat expansion in HTT gene".to_string(),
                required: true,
                test_type: Some(TestType::GeneticSequencing),
                normal_range: Some("<27 CAG repeats".to_string()),
                pathological_range: Some("≥40 CAG repeats".to_string()),
            },
        ],
        differential_diagnosis: vec!["Parkinson's disease".to_string(), "Wilson disease".to_string()],
        genes: vec![
            Gene {
                symbol: "HTT".to_string(),
                name: "Huntingtin".to_string(),
                hgnc_id: "HGNC:4851".to_string(),
                entrez_id: Some(3064),
                ensembl_id: Some("ENSG00000197386".to_string()),
                chromosome: "4".to_string(),
                location: "4p16.3".to_string(),
                function: "Protein involved in vesicular transport and synaptic transmission".to_string(),
                disease_mechanism: DiseaseMechanism::GainOfFunction,
            },
        ],
        phenotypes: Vec::new(),
        icd10_codes: vec!["G10".to_string()],
        icd11_codes: vec!["8A00.0".to_string()],
        omim_codes: vec!["143100".to_string()],
    };

    db.add_disease(huntingtons);

    // Add Cystic Fibrosis
    let cystic_fibrosis = RareDisease {
        orpha_code: "ORPHA:586".to_string(),
        name: "Cystic fibrosis".to_string(),
        synonyms: vec!["CF".to_string(), "Mucoviscidosis".to_string()],
        definition: "A rare genetic disorder affecting the lungs and digestive system".to_string(),
        prevalence: Prevalence {
            point_prevalence: Some(7e-5),
            birth_prevalence: Some(1e-4),
            lifetime_prevalence: None,
            prevalence_class: PrevalenceClass::Rare,
            geographic_distribution: vec!["Worldwide, higher in Caucasians".to_string()],
        },
        inheritance_pattern: vec![InheritancePattern::AutosomalRecessive],
        age_of_onset: vec![AgeOfOnset::Neonatal, AgeOfOnset::Infancy],
        clinical_features: vec![
            ClinicalFeature {
                hpo_id: "HP:0006538".to_string(),
                name: "Recurrent respiratory infections".to_string(),
                frequency: Frequency::VeryFrequent,
                severity: Some(Severity::Severe),
                body_system: BodySystem::Respiratory,
                description: "Chronic lung infections due to thick mucus".to_string(),
            },
            ClinicalFeature {
                hpo_id: "HP:0001508".to_string(),
                name: "Failure to thrive".to_string(),
                frequency: Frequency::Frequent,
                severity: Some(Severity::Moderate),
                body_system: BodySystem::Multiple,
                description: "Poor weight gain and growth".to_string(),
            },
        ],
        diagnostic_criteria: vec![
            DiagnosticCriterion {
                criterion_type: DiagnosticCriterionType::Laboratory,
                description: "Elevated sweat chloride".to_string(),
                required: true,
                test_type: Some(TestType::BloodTest),
                normal_range: Some("<30 mmol/L".to_string()),
                pathological_range: Some(">60 mmol/L".to_string()),
            },
        ],
        differential_diagnosis: vec!["Primary ciliary dyskinesia".to_string(), "Immunodeficiency".to_string()],
        genes: vec![
            Gene {
                symbol: "CFTR".to_string(),
                name: "Cystic fibrosis transmembrane conductance regulator".to_string(),
                hgnc_id: "HGNC:1884".to_string(),
                entrez_id: Some(1080),
                ensembl_id: Some("ENSG00000001626".to_string()),
                chromosome: "7".to_string(),
                location: "7q31.2".to_string(),
                function: "Chloride channel regulating ion transport".to_string(),
                disease_mechanism: DiseaseMechanism::LossOfFunction,
            },
        ],
        phenotypes: Vec::new(),
        icd10_codes: vec!["E84".to_string()],
        icd11_codes: vec!["CA25".to_string()],
        omim_codes: vec!["219700".to_string()],
    };

    db.add_disease(cystic_fibrosis);

    db
}