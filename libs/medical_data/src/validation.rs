use regex::Regex;
use chrono::{DateTime, Utc, NaiveDate};

// Medical data validation functions
pub fn is_valid_date(date_str: &str) -> bool {
    // Support multiple date formats
    let formats = [
        "%Y-%m-%d",
        "%Y/%m/%d", 
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S%.fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ];
    
    for format in &formats {
        if NaiveDate::parse_from_str(date_str, format).is_ok() {
            return true;
        }
        if DateTime::parse_from_rfc3339(date_str).is_ok() {
            return true;
        }
    }
    
    false
}

pub fn is_valid_email(email: &str) -> bool {
    let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
    email_regex.is_match(email)
}

pub fn is_valid_phone(phone: &str) -> bool {
    let phone_regex = Regex::new(r"^\+?[\d\s\-\(\)]{10,}$").unwrap();
    phone_regex.is_match(phone)
}

pub fn is_valid_medical_record_number(mrn: &str) -> bool {
    // Medical record numbers are typically alphanumeric, 6-12 characters
    let mrn_regex = Regex::new(r"^[A-Za-z0-9]{6,12}$").unwrap();
    mrn_regex.is_match(mrn)
}

pub fn is_valid_icd10_code(code: &str) -> bool {
    // ICD-10 codes: Letter followed by 2 digits, optional decimal and 1-4 more digits
    let icd10_regex = Regex::new(r"^[A-Z][0-9]{2}(\.[0-9]{1,4})?$").unwrap();
    icd10_regex.is_match(code)
}

pub fn is_valid_loinc_code(code: &str) -> bool {
    // LOINC codes: 5-6 digits followed by dash and check digit
    let loinc_regex = Regex::new(r"^[0-9]{5,6}-[0-9]$").unwrap();
    loinc_regex.is_match(code)
}

pub fn is_valid_snomed_code(code: &str) -> bool {
    // SNOMED CT codes: 6-18 digits
    let snomed_regex = Regex::new(r"^[0-9]{6,18}$").unwrap();
    snomed_regex.is_match(code)
}

pub fn validate_vital_signs(vital_type: &str, value: f64) -> Result<(), String> {
    match vital_type.to_lowercase().as_str() {
        "temperature_celsius" => {
            if value < 30.0 || value > 45.0 {
                return Err("Temperature out of valid range (30-45°C)".to_string());
            }
        }
        "temperature_fahrenheit" => {
            if value < 86.0 || value > 113.0 {
                return Err("Temperature out of valid range (86-113°F)".to_string());
            }
        }
        "heart_rate" => {
            if value < 20.0 || value > 300.0 {
                return Err("Heart rate out of valid range (20-300 bpm)".to_string());
            }
        }
        "systolic_bp" => {
            if value < 50.0 || value > 300.0 {
                return Err("Systolic BP out of valid range (50-300 mmHg)".to_string());
            }
        }
        "diastolic_bp" => {
            if value < 20.0 || value > 200.0 {
                return Err("Diastolic BP out of valid range (20-200 mmHg)".to_string());
            }
        }
        "respiratory_rate" => {
            if value < 5.0 || value > 60.0 {
                return Err("Respiratory rate out of valid range (5-60 breaths/min)".to_string());
            }
        }
        "oxygen_saturation" => {
            if value < 50.0 || value > 100.0 {
                return Err("Oxygen saturation out of valid range (50-100%)".to_string());
            }
        }
        "weight_kg" => {
            if value < 0.5 || value > 500.0 {
                return Err("Weight out of valid range (0.5-500 kg)".to_string());
            }
        }
        "height_cm" => {
            if value < 30.0 || value > 250.0 {
                return Err("Height out of valid range (30-250 cm)".to_string());
            }
        }
        "bmi" => {
            if value < 10.0 || value > 80.0 {
                return Err("BMI out of valid range (10-80)".to_string());
            }
        }
        _ => {
            return Err(format!("Unknown vital sign type: {}", vital_type));
        }
    }
    
    Ok(())
}

pub fn validate_lab_value(test_name: &str, value: f64, unit: &str) -> Result<(), String> {
    match (test_name.to_lowercase().as_str(), unit.to_lowercase().as_str()) {
        ("glucose", "mg/dl") => {
            if value < 20.0 || value > 800.0 {
                return Err("Glucose out of valid range (20-800 mg/dL)".to_string());
            }
        }
        ("glucose", "mmol/l") => {
            if value < 1.1 || value > 44.4 {
                return Err("Glucose out of valid range (1.1-44.4 mmol/L)".to_string());
            }
        }
        ("hemoglobin", "g/dl") => {
            if value < 3.0 || value > 20.0 {
                return Err("Hemoglobin out of valid range (3-20 g/dL)".to_string());
            }
        }
        ("creatinine", "mg/dl") => {
            if value < 0.1 || value > 15.0 {
                return Err("Creatinine out of valid range (0.1-15 mg/dL)".to_string());
            }
        }
        ("cholesterol", "mg/dl") => {
            if value < 50.0 || value > 500.0 {
                return Err("Cholesterol out of valid range (50-500 mg/dL)".to_string());
            }
        }
        ("white_blood_cells", "k/ul") => {
            if value < 0.5 || value > 100.0 {
                return Err("WBC out of valid range (0.5-100 K/uL)".to_string());
            }
        }
        ("platelets", "k/ul") => {
            if value < 10.0 || value > 2000.0 {
                return Err("Platelets out of valid range (10-2000 K/uL)".to_string());
            }
        }
        _ => {
            // For unknown tests, just check if value is finite
            if !value.is_finite() {
                return Err("Lab value must be a finite number".to_string());
            }
        }
    }
    
    Ok(())
}

pub fn validate_medication_dosage(medication: &str, dose: f64, unit: &str) -> Result<(), String> {
    if dose <= 0.0 {
        return Err("Medication dose must be positive".to_string());
    }
    
    if !dose.is_finite() {
        return Err("Medication dose must be a finite number".to_string());
    }
    
    // Basic unit validation
    let valid_units = [
        "mg", "g", "mcg", "ug", "kg", "ml", "l", "units", "iu", 
        "mg/kg", "mg/m2", "mcg/kg", "units/kg", "drops", "tablets", "capsules"
    ];
    
    if !valid_units.contains(&unit.to_lowercase().as_str()) {
        return Err(format!("Invalid medication unit: {}", unit));
    }
    
    // Medication-specific validation (simplified examples)
    match medication.to_lowercase().as_str() {
        "aspirin" if unit == "mg" => {
            if dose > 1000.0 {
                return Err("Aspirin dose unusually high (>1000mg)".to_string());
            }
        }
        "insulin" if unit == "units" => {
            if dose > 200.0 {
                return Err("Insulin dose unusually high (>200 units)".to_string());
            }
        }
        "morphine" if unit == "mg" => {
            if dose > 100.0 {
                return Err("Morphine dose unusually high (>100mg)".to_string());
            }
        }
        _ => {} // No specific validation for unknown medications
    }
    
    Ok(())
}

pub fn validate_age_consistency(birth_date: &str, reported_age: Option<u32>) -> Result<(), String> {
    if !is_valid_date(birth_date) {
        return Err("Invalid birth date format".to_string());
    }
    
    if let Some(age) = reported_age {
        // Calculate age from birth date
        let birth = NaiveDate::parse_from_str(birth_date, "%Y-%m-%d")
            .map_err(|_| "Could not parse birth date")?;
        
        let today = Utc::now().date_naive();
        let calculated_age = today.years_since(birth).unwrap_or(0);
        
        // Allow for some variance (±1 year) due to exact date differences
        if (calculated_age as i32 - age as i32).abs() > 1 {
            return Err(format!(
                "Age inconsistency: calculated {} years from birth date, reported {} years",
                calculated_age, age
            ));
        }
    }
    
    Ok(())
}

pub fn validate_medical_identifier_checksum(identifier_type: &str, identifier: &str) -> Result<(), String> {
    match identifier_type.to_lowercase().as_str() {
        "npi" => validate_npi_checksum(identifier),
        "ssn" => validate_ssn_format(identifier),
        "medicare" => validate_medicare_format(identifier),
        _ => Ok(()), // No validation for unknown identifier types
    }
}

fn validate_npi_checksum(npi: &str) -> Result<(), String> {
    if npi.len() != 10 {
        return Err("NPI must be exactly 10 digits".to_string());
    }
    
    if !npi.chars().all(|c| c.is_ascii_digit()) {
        return Err("NPI must contain only digits".to_string());
    }
    
    // Luhn algorithm for NPI validation
    let digits: Vec<u32> = npi.chars()
        .map(|c| c.to_digit(10).unwrap())
        .collect();
    
    let mut sum = 0;
    for (i, &digit) in digits.iter().enumerate() {
        let mut d = digit;
        if i % 2 == 1 {
            d *= 2;
            if d > 9 {
                d = d / 10 + d % 10;
            }
        }
        sum += d;
    }
    
    if sum % 10 != 0 {
        return Err("Invalid NPI checksum".to_string());
    }
    
    Ok(())
}

fn validate_ssn_format(ssn: &str) -> Result<(), String> {
    let ssn_regex = Regex::new(r"^\d{3}-\d{2}-\d{4}$").unwrap();
    if !ssn_regex.is_match(ssn) {
        return Err("SSN must be in format XXX-XX-XXXX".to_string());
    }
    
    // Check for invalid SSN patterns
    let parts: Vec<&str> = ssn.split('-').collect();
    if parts[0] == "000" || parts[0] == "666" || parts[0].starts_with('9') {
        return Err("Invalid SSN area number".to_string());
    }
    
    if parts[1] == "00" {
        return Err("Invalid SSN group number".to_string());
    }
    
    if parts[2] == "0000" {
        return Err("Invalid SSN serial number".to_string());
    }
    
    Ok(())
}

fn validate_medicare_format(medicare: &str) -> Result<(), String> {
    // Medicare Beneficiary Identifier (MBI) format: 1A2A3A4A5A6
    let mbi_regex = Regex::new(r"^[1-9][A-Z][0-9][A-Z][0-9][A-Z][0-9][A-Z][0-9][A-Z][0-9]$").unwrap();
    if mbi_regex.is_match(medicare) {
        return Ok(());
    }
    
    // Legacy Medicare format: AAANNNNA
    let legacy_regex = Regex::new(r"^[A-Z]{3}[0-9]{4}[A-Z]$").unwrap();
    if legacy_regex.is_match(medicare) {
        return Ok(());
    }
    
    Err("Invalid Medicare identifier format".to_string())
}

pub fn validate_clinical_data_consistency(
    observations: &[crate::Observation],
    conditions: &[crate::Condition],
) -> Result<Vec<String>, String> {
    let mut warnings = Vec::new();
    
    // Check for conflicting observations
    for obs1 in observations {
        for obs2 in observations {
            if obs1.id != obs2.id && obs1.subject.reference == obs2.subject.reference {
                if let (Some(date1), Some(date2)) = (&obs1.effective_datetime, &obs2.effective_datetime) {
                    if date1 == date2 && obs1.code.coding == obs2.code.coding {
                        if let (Some(val1), Some(val2)) = (&obs1.value, &obs2.value) {
                            // Check for significantly different values for same test on same date
                            if let (crate::ObservationValue::Quantity(q1), crate::ObservationValue::Quantity(q2)) = (val1, val2) {
                                if let (Some(v1), Some(v2)) = (q1.value, q2.value) {
                                    let diff_percent = ((v1 - v2).abs() / v1.max(v2)) * 100.0;
                                    if diff_percent > 50.0 {
                                        warnings.push(format!(
                                            "Conflicting values for same test on same date: {} vs {}",
                                            v1, v2
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Check for observations that don't match conditions
    for condition in conditions {
        if let Some(ref condition_code) = condition.code {
            let related_observations: Vec<&crate::Observation> = observations
                .iter()
                .filter(|obs| obs.subject.reference == condition.subject.reference)
                .collect();
            
            // This is a simplified check - in practice would use medical ontologies
            if condition_code.text.as_ref().map_or(false, |text| text.to_lowercase().contains("diabetes")) {
                let has_glucose_test = related_observations
                    .iter()
                    .any(|obs| {
                        obs.code.text.as_ref().map_or(false, |text| text.to_lowercase().contains("glucose"))
                    });
                
                if !has_glucose_test {
                    warnings.push(format!(
                        "Diabetes condition without glucose test for patient {}",
                        condition.subject.reference.as_ref().unwrap_or(&"unknown".to_string())
                    ));
                }
            }
        }
    }
    
    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_validation() {
        assert!(is_valid_date("2023-12-25"));
        assert!(is_valid_date("2023/12/25"));
        assert!(is_valid_date("25-12-2023"));
        assert!(is_valid_date("2023-12-25T10:30:00Z"));
        assert!(!is_valid_date("2023-13-25")); // Invalid month
        assert!(!is_valid_date("invalid-date"));
    }

    #[test]
    fn test_email_validation() {
        assert!(is_valid_email("test@example.com"));
        assert!(is_valid_email("user.name+tag@domain.co.uk"));
        assert!(!is_valid_email("invalid-email"));
        assert!(!is_valid_email("@domain.com"));
    }

    #[test]
    fn test_vital_signs_validation() {
        assert!(validate_vital_signs("temperature_celsius", 37.0).is_ok());
        assert!(validate_vital_signs("heart_rate", 72.0).is_ok());
        assert!(validate_vital_signs("temperature_celsius", 50.0).is_err());
        assert!(validate_vital_signs("heart_rate", 400.0).is_err());
    }

    #[test]
    fn test_icd10_validation() {
        assert!(is_valid_icd10_code("A00"));
        assert!(is_valid_icd10_code("Z99.9"));
        assert!(is_valid_icd10_code("M79.3"));
        assert!(!is_valid_icd10_code("999"));
        assert!(!is_valid_icd10_code("A"));
    }

    #[test]
    fn test_npi_validation() {
        // This is a test NPI with valid checksum
        assert!(validate_npi_checksum("1234567893").is_ok());
        assert!(validate_npi_checksum("123456789").is_err()); // Wrong length
        assert!(validate_npi_checksum("123456789a").is_err()); // Contains letter
    }
}