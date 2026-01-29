#pragma once

#include "llama.h"
#include "llama-model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Engram configuration structure
struct engram_config {
    int max_ngram_size;
    int n_embed_per_ngram;
    int n_head_per_ngram;
    int * layer_ids; // array of layer IDs where engram should be applied
    int n_layer_ids;
    int pad_id;
    int seed;
    int kernel_size;
    const char * tokenizer_name_or_path;
};

// Engram context structure for runtime
struct engram_context {
    struct engram_config cfg;
    
    // Hash mapping and embedding tables would go here
    // This is a simplified version - actual implementation 
    // will need to manage these more carefully
    
    bool initialized;
};

// Create an engram context
struct engram_context * engram_init(struct engram_config cfg);

// Free an engram context
void engram_free(struct engram_context * ctx);

// Apply engram to a layer
ggml_tensor * engram_apply(
    struct engram_context * ctx,
    ggml_context * ctx0,
    const llama_model & model,
    ggml_tensor * hidden_states, 
    ggml_tensor * input_ids,
    int layer_id);

#ifdef __cplusplus
}
#endif