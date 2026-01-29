# Engram Integration for llama.cpp

This document describes the integration of Engram into the llama.cpp framework.

## What is Engram?

Engram is a method that enhances transformer models by incorporating explicit N-gram features directly into the model architecture. It works by:

1. **N-gram Hashing**: Computing hash values from consecutive token sequences
2. **Context-aware Gating**: Using hidden states as queries to gate the N-gram features
3. **Short Convolution**: Applying learned convolution operations on the hashed features
4. **Residual Connection**: Adding the processed features back to the original hidden states

## Integration Approach

The Engram module is integrated into llama.cpp by:

1. Creating `llama-engram.h` and `llama-engram.cpp` files that provide the core functionality
2. Adding these files to the CMake build system
3. Providing a clean API for configuration and application

## Key Components

### 1. Configuration Structure (`engram_config`)
```c
struct engram_config {
    int max_ngram_size;        // Maximum N-gram size to consider
    int n_embed_per_ngram;     // Embedding dimension per N-gram
    int n_head_per_ngram;      // Number of attention heads per N-gram
    int * layer_ids;           // Array of layer IDs where Engram should be applied
    int n_layer_ids;           // Number of layers in the layer_ids array
    int pad_id;                // Padding token ID
    int seed;                  // Random seed for hashing
    int kernel_size;           // Size of convolutional kernel
    const char * tokenizer_name_or_path; // Tokenizer path for vocabulary access
};
```

### 2. Runtime Context (`engram_context`)
```c
struct engram_context {
    struct engram_config cfg;
    bool initialized;
};
```

### 3. Core Function
```c
ggml_tensor * engram_apply(
    struct engram_context * ctx,
    ggml_context * ctx0,
    const llama_model & model,
    ggml_tensor * hidden_states, 
    ggml_tensor * input_ids,
    int layer_id);
```

## Integration Points

Engram can be applied at specific layers in the transformer architecture. The typical integration point is after the feed-forward network (FFN) but before the residual connection:

```cpp
// In the LLaMA model building loop:
cur = build_ffn(cur, ...); // FFN computation
// Apply Engram here:
cur = engram_apply(engram_ctx, ctx0, model, cur, input_ids, il);
cur = ggml_add(ctx0, cur, ffn_inp); // Residual connection
```

## Usage Example

```c
// Initialize Engram
struct engram_config cfg = {
    .max_ngram_size = 3,
    .n_embed_per_ngram = 128,
    .n_head_per_ngram = 4,
    .layer_ids = new int[2]{0, 1}, // Apply to layers 0 and 1
    .n_layer_ids = 2,
    .pad_id = 0,
    .seed = 42,
    .kernel_size = 3,
    .tokenizer_name_or_path = "tokenizer.json"
};

struct engram_context * engram_ctx = engram_init(cfg);

// In model processing loop:
cur = engram_apply(engram_ctx, ctx0, model, cur, input_ids, layer_id);

// Clean up
engram_free(engram_ctx);
delete[] cfg.layer_ids;
```

## Build Instructions

The Engram module is automatically included in the llama.cpp build system. To build:

```bash
cd llama.cpp
mkdir -p build
cd build
cmake ..
make
```

## Implementation Notes

1. The current implementation provides a framework and placeholder functions
2. Full N-gram hashing, convolution, and gating operations need to be implemented in `llama-engram.cpp`
3. Memory management follows llama.cpp conventions using ggml tensors
4. The integration is designed to be modular and configurable for different model architectures

## Future Work

1. Implement full N-gram hashing algorithm
2. Add convolutional layer for short context processing  
3. Implement attention-based gating mechanism
4. Add proper memory management for hash tables and embeddings
5. Optimize performance for large-scale models