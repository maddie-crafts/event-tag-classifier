# Event Tag Classifier

**Event Tag Classifier** is a lightweight Rust-based text classification system designed to identify and label short social event descriptions with tags like `festival`, `conference`, `party/night out`, and more. It is built using the [Burn](https://burn.dev/) deep learning framework and supports inference in both native Rust and WebAssembly environments.

---

## ✨ Features

- Classifies short sentences into 7 predefined event categories
- Built with [Burn v0.16](https://github.com/tracel-ai/burn), a blazing-fast deep learning framework in Rust
- Includes a custom model architecture with embeddings and fully connected layers
- [TO DO] WebAssembly-ready for running in the browser
- [TO DO] CLI and WASM interface for flexible integration