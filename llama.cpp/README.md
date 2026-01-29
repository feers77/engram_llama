# 🧠 llama.cpp with Engram Integration

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ggml-org/llama.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/build.yml)

A powerful, efficient implementation of LLM inference in C/C++ with **Engram integration** for enhanced contextual understanding.

## 📖 Overview

This repository contains a modified version of llama.cpp with **Engram integration**, designed to enhance transformer models by incorporating explicit N-gram features directly into the model architecture. Engram technology provides improved context awareness and memory representation capabilities.

### Key Features
- ✅ **Engram Integration**: Enhanced contextual processing through N-gram feature incorporation
- ✅ **gpt-oss:120 Model Support**: Ready to run the powerful gpt-oss:120b model
- ✅ **SSL Support**: Secure Hugging Face model downloads
- ✅ **Cross-platform**: Works on Linux, macOS, and Windows
- ✅ **Multi-architecture**: Supports x86, ARM, and other CPU architectures

## 🚀 Quick Start

### Prerequisites
- C++17 compiler (GCC 9+, Clang 9+, MSVC 2019+)
- CMake 3.14+
- OpenSSL development libraries (for model downloads)

### Installation
```bash
# Clone the repository
git clone https://github.com/feers77/engram_llama.git
cd Engram/llama.cpp

# Install SSL dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install libssl-dev

# Build with SSL support
mkdir -p build
cd build
cmake -DLLAMA_OPENSSL=ON ..
make -j$(nproc)
```

### Running the gpt-oss:120 Model
```bash
# Run with a simple prompt
./bin/llama-cli --gpt-oss-120b-default --prompt "Hello, how are you?"

# Start interactive session
./bin/llama-cli --gpt-oss-120b-default

# Run with custom parameters
./bin/llama-cli --gpt-oss-120b-default \
  --threads 8 \
  --ctx-size 2048 \
  --temp 0.7 \
  --prompt "Explain quantum computing"
```

## 📁 Model Location & Management

### Model Storage
Models are automatically downloaded and cached in:
```bash
~/.cache/llama.cpp/
```

### Available Models
The system supports:
- `ggml-org/gpt-oss-120b-GGUF` (gpt-oss:120 model)
- Other GGUF-compatible models from Hugging Face

### Manual Model Download
If you need to manually download a model:
```bash
# Using Hugging Face CLI (if installed)
huggingface-cli download ggml-org/gpt-oss-120b-GGUF --local-dir ~/.cache/llama.cpp/

# Or via curl/wget (for specific files)
curl -L -o ~/.cache/llama.cpp/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-mxfp4-00001-of-00003.gguf
```

## 📁 File Structure

### Core Engram Files
```
llama.cpp/
├── src/
│   ├── llama-engram.h          # Engram interface header
│   ├── llama-engram.cpp        # Engram implementation
│   └── llama-engram-complete.cpp # Complete Engram integration
├── include/
│   └── llama-engram.h          # Public API header
├── examples/
│   ├── engram-integration-example.cpp  # Engram usage example
│   └── engram-load-model-example.cpp   # Model loading with Engram
└── README_ENGRAM.md            # Detailed Engram documentation
```

## 🔧 Configuration & Parameters

### Basic Usage
```bash
# Default gpt-oss:120b model
./bin/llama-cli --gpt-oss-120b-default --prompt "Your question here"

# With custom parameters
./bin/llama-cli --gpt-oss-120b-default \
  -t 8          # Number of threads
  -c 2048       # Context size
  -n 256        # Max tokens to predict
  --temp 0.7    # Temperature
  --top-p 0.9   # Top-P sampling
```

### Engram-Specific Options
```bash
# Engram configuration (if custom)
./bin/llama-cli --gpt-oss-120b-default \
  --engram-max-ngram-size 3 \
  --engram-n-embed-per-ngram 128
```

## 📦 Files to Upload for Open Source

### Essential Files for Repository
The following files are needed for your open-source repository:

#### Core Implementation Files
```bash
# Engram integration source files
llama.cpp/src/llama-engram.cpp
llama.cpp/src/llama-engram-complete.cpp
llama.cpp/include/llama-engram.h

# Header files
llama.cpp/src/llama-engram.h
```

#### Examples and Documentation
```bash
# Example usage files
llama.cpp/examples/engram-integration-example.cpp
llama.cpp/examples/engram-load-model-example.cpp

# Documentation
llama.cpp/README_ENGRAM.md
llama.cpp/GPT-OSS-120-TUTORIAL.md
```

#### Build Configuration
```bash
# Build files that need to be included
llama.cpp/CMakeLists.txt
llama.cpp/cmake/
llama.cpp/Makefile
llama.cpp/build-xcframework.sh
```

#### License and Metadata
```bash
# Required license files
llama.cpp/LICENSE
llama.cpp/README.md
llama.cpp/README_ENGRAM.md
llama.cpp/CONTRIBUTING.md
llama.cpp/CODEOWNERS
llama.cpp/SECURITY.md
```

### Files to Exclude from Repository
The following should **NOT** be included in your repository:
- Model files (too large, ~20GB for gpt-oss:120)
- Build artifacts (`build/` directory)
- Cache directories (`~/.cache/llama.cpp/`)
- Generated binaries (`bin/` directory)

## 📊 Performance & Requirements

### System Requirements
- **RAM**: 16GB+ (32GB recommended for optimal performance)
- **Storage**: 50GB+ of free disk space (for models and cache)
- **CPU**: Modern multi-core processor (Intel/AMD ARM)
- **GPU**: Optional but recommended for faster inference

### Performance Tips
```bash
# Optimize for speed
./bin/llama-cli --gpt-oss-120b-default \
  -t 16 \          # Use more threads
  -c 4096 \        # Increase context
  --temp 0.3       # Lower temperature for faster, more deterministic responses

# Optimize for quality  
./bin/llama-cli --gpt-oss-120b-default \
  -t 8 \           # Use fewer threads to avoid overhead
  -c 2048 \        # Standard context size
  --temp 0.7 \     # Higher temperature for creative responses
  --top-p 0.9      # Better sampling diversity
```

## 🛠️ Troubleshooting

### SSL Errors
If you encounter SSL-related issues:
```bash
# Rebuild with proper SSL support
cd llama.cpp/build
cmake -DLLAMA_OPENSSL=ON ..
make -j$(nproc)
```

### Model Download Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/llama.cpp/
# Then run your command again
./bin/llama-cli --gpt-oss-120b-default --prompt "Hello"
```

## 📚 Resources

### Documentation
- [Engram Integration Documentation](README_ENGRAM.md)
- [gpt-oss:120 Tutorial](GPT-OSS-120-TUTORIAL.md)
- [llama.cpp Official Docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

### Related Projects
- [Hugging Face GGUF Models](https://huggingface.co/models?library=gguf&sort=trending)
- [Engram GitHub Repository](https://github.com/deepseek-ai/Engram)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Based on [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Engram implementation inspired by [DeepSeek AI research](https://github.com/deepseek-ai/Engram)
- Built with contributions from the open-source community

---

*Powered by llama.cpp with Engram integration*
