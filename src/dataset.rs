
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use serde_json::Result;

#[derive(Debug, Deserialize)]
struct CategoryData {
    valid_domain: Vec<String>,
    invalid_domain: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DataFile {
    categories: HashMap<String, CategoryData>,
}

#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub text: String,
    pub label: usize,
    pub is_valid: bool
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub examples: Vec<TrainingExample>,
    pub categories: HashMap<usize, String>
}

impl Dataset {
    pub fn from_json(path: &str) -> Result<Self> {
        let file_content = fs::read_to_string(path).expect("Failed to read the data file");
        let data_file: DataFile = serde_json::from_str(&file_content)?;

        let mut examples = Vec::new();
        let mut categories = HashMap::new();
        let mut category_id = 0;

        for (category_name, data) in data_file.categories {
            categories.insert(category_id, category_name);

            for domain in data.valid_domain {
                examples.push(TrainingExample {
                    text: domain,
                    label: category_id,
                    is_valid: true
                });
            }

            for domain in data.invalid_domain {
                examples.push(TrainingExample {
                    text: domain,
                    label: category_id,
                    is_valid: false
                });
            }

            category_id += 1;
        }

        Ok(Dataset {
            examples,
            categories
        })

    }
}