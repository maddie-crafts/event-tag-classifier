mod dataset;
mod vocabulary;
mod preprocess;
mod model;

use std::{fs::write, path::Path};

use burn::{
    backend::wgpu::{JitBackend, WgpuDevice, WgpuRuntime},
    module::Module,
    nn::loss::CrossEntropyLoss,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, PrettyJsonFileRecorder},
    tensor::{activation::softmax, Int, Tensor},
};
use burn_autodiff::Autodiff;
use burn_fusion::Fusion;
use dataset::Dataset;
use model::TextClassifier;
use vocabulary::Vocabulary;
use serde_json;

type Inner = Fusion<JitBackend<WgpuRuntime, f32, i32, u32>>;
type Backend = Autodiff<Inner>;

const CATEGORIES: [&str; 7] = [
    "festival",
    "sporting event",
    "family meal",
    "outing with friends",
    "party/night out",
    "ceremony",
    "car gathering",
];

#[derive(Debug)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
}

fn main() {
    let model_path = "model.json";
    let mut model = if Path::new(model_path).exists() {
        unimplemented!("Model loading not yet implemented.");
    } else {
        let dataset = Dataset::from_json("dataset.json").expect("Failed to load dataset");
        let mut vocab = Vocabulary::new();
        let max_len = 10;
        let preprocessed = preprocess::preprocess_dataset(dataset, &mut vocab, max_len);

        let vocab_json = serde_json::to_string_pretty(&vocab).expect("Failed to serialize vocab");
        write("vocab.json", vocab_json).expect("Failed to write vocab.json");
        println!("Saved frozen vocabulary to vocab.json");

        let vocab_size = vocab.size();
        let embedding_dim = 128;
        let hidden_dim = 64;
        let num_classes = CATEGORIES.len();

        let device = WgpuDevice::default();
        let mut model = TextClassifier::<Backend>::new(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_classes,
            &device,
        );

        let mut optimizer = AdamConfig::new().init::<Backend, _>();
        let criterion = CrossEntropyLoss::new(None, &device);

        let config = TrainingConfig {
            epochs: 15,
            learning_rate: 0.001,
        };

        for epoch in 0..config.epochs {
            println!("Epoch {epoch}");
            for example in &preprocessed {
                let tokens: [i32; 10] = example
                    .tokens
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("Token length must be exactly 10");

                let input = Tensor::<Backend, 2, Int>::from_data([tokens], &device);
                let target = Tensor::<Backend, 1, Int>::from_data([example.label], &device);

                let output = model.forward(input);
                let loss = criterion.forward(output, target);

                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(config.learning_rate, model, grads);
            }
        }

        let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        model.clone()
            .save_file::<_, _>("model.json", &recorder)
            .expect("Failed to save model");

        model
    };

    println!("Model is ready for inference.");

    let vocab_json = std::fs::read_to_string("vocab.json").expect("Failed to read vocab.json");
    let mut vocab: Vocabulary = serde_json::from_str(&vocab_json).expect("Invalid vocab format");

    // Test prediction
    let test_sentence = "The music festival is in the park";
    let tokens = vocab.encode_sentence(test_sentence);
    let padded_tokens: [i32; 10] = tokens
        .into_iter()
        .map(|x| x as i32)
        .chain(std::iter::repeat(0))
        .take(10)
        .collect::<Vec<_>>()
        .try_into()
        .expect("Expected exactly 10 tokens");

    let device = WgpuDevice::default();
    let input = Tensor::<Backend, 2, Int>::from_data([padded_tokens], &device);
    let output = model.forward(input);
    let probs = softmax(output, 1);
    let probs_data = probs.into_data();
    let slice: &[f32] = probs_data.as_slice().expect("Failed to get tensor data slice");
    let predicted_idx = slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .expect("Failed to get max index");

    println!("Predicted category: {}", CATEGORIES[predicted_idx]);
}