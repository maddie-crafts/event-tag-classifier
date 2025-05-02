use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Relu, Initializer};
use burn::tensor::{Tensor, Int, backend::Backend};

#[derive(Debug, burn::module::Module)]
pub struct TextClassifier<B: Backend> {
    embedding: Embedding<B>,
    fc1: Linear<B>,
    relu: Relu,
    output: Linear<B>,
}

impl<B: Backend> TextClassifier<B> {
    pub fn new(vocab_size: usize, embedding_dim: usize, hidden_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, embedding_dim)
            .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 })
            .init(device);

        let fc1 = LinearConfig::new(embedding_dim, hidden_dim)
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);

        let output = LinearConfig::new(hidden_dim, num_classes)
            .with_initializer(Initializer::XavierUniform {
                gain: 1.0,
            })
            .init(device);

        Self {
            embedding,
            fc1,
            relu: Relu::new(),
            output,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let x = self.embedding.forward(input);   // [batch, seq_len, embed_dim]
        let x = x.mean_dim(1).squeeze(1);                  // [batch, embed_dim]
        let x = self.fc1.forward(x);            // [batch, hidden_dim]
        let x = self.relu.forward(x);
        self.output.forward(x)                  // [batch, num_classes]
    }
}
