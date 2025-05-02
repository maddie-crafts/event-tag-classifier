use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Vocabulary {
    pub word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
}

impl Vocabulary {

    pub fn new() -> Self {
        Vocabulary {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
        }
    }

    pub fn add_word(&mut self, word: &str) -> usize {
        if let Some(&id) = self.word_to_id.get(word) {
            id
        } else {
            let id = self.id_to_word.len();
            self.word_to_id.insert(word.to_string(), id);
            self.id_to_word.push(word.to_string());
            id
        }
    }

    pub fn encode_sentence(&mut self, sentence: &str) -> Vec<usize> {
        sentence
            .split_whitespace()
            .map(|word| self.add_word(word))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.id_to_word.len()
    }

}