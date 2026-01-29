// Example of loading a model with Engram integration
// This demonstrates how to use the Engram functionality with a real model

#include "llama.h"
#include "llama-engram.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }
    
    const char * model_path = argv[1];
    
    // Initialize llama.cpp backend
    llama_backend_init();
    
    // Model parameters
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // Set to -1 for all layers on GPU if available
    
    // Load the model
    printf("Loading model from %s...\n", model_path);
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    
    if (model == nullptr) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }
    
    // Context parameters
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048; // Set context size
    
    // Initialize the context
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    
    if (ctx == nullptr) {
        fprintf(stderr, "Error: failed to initialize context\n");
        llama_model_free(model);
        return 1;
    }
    
    printf("Model loaded successfully!\n");
    printf("Model info:\n");
    printf("  Context size: %d\n", llama_n_ctx(ctx));
    printf("  Embedding size: %d\n", llama_model_n_embd(model));
    printf("  Number of layers: %d\n", llama_model_n_layer(model));
    
    // Initialize Engram
    struct engram_config cfg = {
        .max_ngram_size = 3,
        .n_embed_per_ngram = 128,
        .n_head_per_ngram = 4,
        .layer_ids = new int[2]{0, 1}, // Apply to first two layers
        .n_layer_ids = 2,
        .pad_id = 0,
        .seed = 42,
        .kernel_size = 3,
        .tokenizer_name_or_path = nullptr
    };
    
    struct engram_context * engram_ctx = engram_init(cfg);
    
    if (engram_ctx == nullptr) {
        fprintf(stderr, "Error: failed to initialize Engram\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    printf("Engram initialized successfully!\n");
    
    // Example usage - tokenize input
    const char * prompt = "Hello, how are you?";
    std::vector<llama_token> tokens;
    
    int32_t n_tokens = llama_tokenize(llama_model_get_vocab(model), 
                                      prompt, strlen(prompt), 
                                      tokens.data(), tokens.size(), true, false);
    
    if (n_tokens < 0) {
        fprintf(stderr, "Error: failed to tokenize prompt\n");
        engram_free(engram_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    printf("Prompt tokenized successfully!\n");
    
    // Clean up
    engram_free(engram_ctx);
    delete[] cfg.layer_ids;
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    printf("Model and Engram cleaned up successfully!\n");
    
    return 0;
}