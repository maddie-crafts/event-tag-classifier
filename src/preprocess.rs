use crate::{dataset::Dataset, vocabulary::Vocabulary};

#[derive(Debug, Clone)]
pub struct PreprocessedExample {
    pub tokens: Vec<usize>,
    pub label: usize,
    pub is_valid: bool,
}

pub fn preprocess_dataset(dataset: Dataset, vocab: &mut Vocabulary, max_len: usize) -> Vec<PreprocessedExample> {
    dataset
        .examples
        .into_iter()
        .map(|example| {
            let mut tokens = vocab.encode_sentence(&example.text);
            tokens.resize(max_len, 0);
            PreprocessedExample {
                tokens,
                label: example.label,
                is_valid: example.is_valid,
            }
        })
        .collect()
}