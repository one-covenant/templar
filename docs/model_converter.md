# Templar Model Converter

The Templar Model Converter is an autonomous service that continuously monitors model checkpoints, saves them with proper versioning, converts them to GGUF format, and optionally uploads them to deployment platforms. It automatically detects new checkpoints and performs conversions with optional uploads on a configurable interval.

## Overview

The model converter script (`scripts/model_converter.py`) performs:
- Automatic checkpoint detection by window number
- Versioned model saving using semver-compatible format
- GGUF format conversion for deployment
- Optional upload to HuggingFace Hub and Ollama
- Resource management and cleanup
- Service-oriented design for continuous operation

## Features

- **Automated Monitoring**: Continuously checks for new model checkpoints
- **Semantic Versioning**: Saves models with format `{version}-alpha+{global_step}`
- **GGUF Conversion**: Converts models to GGUF format for efficient inference
- **HuggingFace Hub Integration**: Automatic upload to HuggingFace repositories
- **Ollama Integration**: Direct deployment to local Ollama installation
- **Flexible Authentication**: Support for environment variables and configuration
- **Resource Management**: Cleans up previous model versions automatically
- **Configurable Intervals**: Customizable conversion and upload frequency
- **Error Handling**: Robust error handling for conversion and upload failures

## Prerequisites

### System Requirements

- Registered Bittensor wallet
- NVIDIA GPU with CUDA support
- Python 3.12+
- uv package manager
- Internet connection (for downloading conversion script)
- Sufficient disk space for model storage

### Environment Variables

The converter requires:
- Standard Templar environment configuration
- Access to model checkpoints on the network

For upload functionality (optional):
- `HF_TOKEN`: HuggingFace authentication token for Hub uploads
- Ollama service running locally for Ollama integration

## Installation

The converter automatically handles its dependencies:

1. **Automatic Script Download**: Downloads the GGUF conversion script if not present
2. **Dependency Installation**: Installs the `gguf` Python package if missing
3. **Upload Dependencies**: Installs `huggingface_hub` for HuggingFace uploads if needed
4. **Directory Creation**: Creates necessary directories for model storage

These steps happen automatically when you first run the converter.

For upload functionality, install optional dependencies:
```bash
pip install huggingface_hub  # For HuggingFace uploads
curl -fsSL https://ollama.ai/install.sh | sh  # For Ollama integration
```

## Usage

### Basic Execution

Run the converter with default settings:
```bash
uv run ./scripts/model_converter.py
```

### Custom Configuration

Run with specific parameters:
```bash
uv run scripts/model_converter.py \
  --netuid 3 \
  --device cuda:0 \
  --conversion_interval 300 \
  --checkpoint_path custom/checkpoints/
```

### With Upload Integration

Run with automatic uploads enabled:
```bash
# Set authentication
export HF_TOKEN=hf_your_token_here

# Run with uploads enabled
uv run scripts/model_converter.py \
  --netuid 3 \
  --device cuda:0 \
  --upload-hf \
  --upload-ollama \
  --hf_repo_id "myuser/templar-models"
```

### Command-Line Arguments

#### Core Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--netuid` | 3 | Bittensor network UID |
| `--device` | cuda:1 | GPU device to use for model loading |
| `--conversion_interval` | 600 | Seconds between conversion checks (10 minutes) |
| `--checkpoint_path` | checkpoints/ | Directory for checkpoint storage |
| `--uid` | None | Override the wallet's UID |

#### Upload Arguments (Future Implementation)

| Argument | Default | Description |
|----------|---------|-------------|
| `--upload-hf` | False | Upload converted models to HuggingFace Hub |
| `--upload-ollama` | False | Upload GGUF models to Ollama |
| `--hf_repo_id` | Auto-generated | HuggingFace repository ID |
| `--hf_token` | HF_TOKEN env | HuggingFace authentication token |
| `--ollama_model_name` | Auto-generated | Custom Ollama model name |
| `--private` | False | Create private HuggingFace repository |

## Architecture

### Conversion Workflow

1. **Checkpoint Detection**: Monitors blockchain for new checkpoints by window number
2. **Model Loading**: Downloads and loads checkpoint when new version is detected
3. **Versioned Saving**: Saves model with semantic version format
4. **GGUF Conversion**: Converts model to GGUF format using conversion script
5. **Upload to Services**: Optionally uploads to HuggingFace Hub and/or Ollama
6. **Cleanup**: Deletes previous model versions to save space
7. **Metrics Logging**: Reports conversion and upload metrics to monitoring systems
8. **Wait**: Sleeps until next conversion interval

### Key Components

- **ModelConverter Class**: Main orchestrator for the conversion process
- **CommsClass**: Handles network communication and checkpoint retrieval
- **MetricsLogger**: Manages metrics submission for monitoring
- **MetricsLoggerWriteOptions**: Configures batching for InfluxDB writes
- **GGUF Script Integration**: Uses official llama.cpp conversion script
- **Upload Integrations**: HuggingFace Hub and Ollama deployment handlers
- **Authentication Manager**: Handles tokens and credentials for uploads

### Versioning Format

The converter uses a semantic versioning format:
```
{package_version}-alpha+{global_step}
```

Examples:
- `0.2.84-alpha+1000`
- `0.2.85-alpha+2500`

This format is:
- Compatible with semantic versioning standards
- Sortable by version and step
- Clear about the model's training progress

## File Structure

The converter creates the following structure:
```
models/
└── upload/
    ├── 0.2.84-alpha+1000/
    │   ├── config.json
    │   ├── model.gguf
    │   ├── pytorch_model.bin
    │   └── tokenizer files...
    └── 0.2.84-alpha+2000/
        └── ... (latest version)
```

Only the latest version is retained; previous versions are automatically deleted.

## GGUF Conversion

### What is GGUF?

GGUF (GGML Universal Format) is an efficient model format that:
- Enables fast inference on various hardware
- Reduces model size through quantization
- Supports CPU and GPU inference
- Is widely compatible with inference engines

### Conversion Process

1. **Script Download**: Automatically downloads `convert_hf_to_gguf.py` from llama.cpp
2. **Dependency Check**: Ensures `gguf` Python package is installed
3. **Conversion**: Runs the conversion script on the saved model
4. **Validation**: Verifies the GGUF file exists after conversion

### Conversion Command

The converter executes:
```bash
uv run python scripts/convert_hf_to_gguf.py {model_dir}/ --outfile {model_dir}/model.gguf
```

## Monitoring

### Metrics Reported

The converter reports metrics including:
- `conversion_timestamp`: When the conversion occurred
- `gguf_converted`: Success indicator (1.0 for success)
- `hf_uploaded`: HuggingFace upload success indicator
- `ollama_uploaded`: Ollama upload success indicator
- `global_step`: Training step of the converted model
- `window`: Window number of the checkpoint
- `version`: Semantic version string

### Viewing Metrics

Access metrics through your monitoring dashboard to track:
- Conversion success rates
- Upload success rates
- Model version history
- Processing and upload times
- System health

## Troubleshooting

### Common Issues

1. **Conversion Script Not Found**:
   - The script will be automatically downloaded
   - Check internet connectivity
   - Verify write permissions in scripts directory

2. **GGUF Package Missing**:
   - The package will be automatically installed
   - Ensure pip is properly configured
   - Check Python environment

3. **Disk Space Issues**:
   - Monitor available disk space
   - Previous versions are auto-deleted
   - Adjust conversion interval if needed

4. **GPU Memory Errors**:
   - Ensure GPU has sufficient memory
   - Check for other processes using GPU
   - Consider using a different GPU device

5. **Conversion Failures**:
   - Check conversion script output
   - Verify model format compatibility
   - Review error logs for details

6. **Upload Failures**:
   - Verify HF_TOKEN is set and valid
   - Check internet connectivity
   - Ensure Ollama service is running
   - Review authentication credentials
   - Check repository permissions

### Debug Mode

Enable detailed logging:
```bash
export RUST_LOG=debug
uv run scripts/model_converter.py
```

## Best Practices

1. **Resource Management**:
   - Monitor disk space usage
   - Ensure adequate GPU memory
   - Use appropriate conversion intervals

2. **Scheduling**:
   - Set intervals based on checkpoint frequency
   - Consider system load when scheduling
   - Avoid overlapping with other intensive tasks

3. **Monitoring**:
   - Set up alerts for conversion failures
   - Track disk space usage
   - Monitor conversion success rates

4. **Deployment**:
   - Use dedicated storage for models
   - Implement backup strategies
   - Plan for model distribution

## Integration

The model converter integrates with:
- The same wallet system as miners/validators
- Checkpoint distribution network
- Monitoring infrastructure
- Version control systems
- HuggingFace Hub for model sharing
- Ollama for local model deployment
- CI/CD pipelines for automated deployment

## Advanced Usage

### Custom Conversion Scripts

To use a different conversion script:
1. Modify `GGUF_SCRIPT_PATH` in the code
2. Ensure script compatibility
3. Test thoroughly before production use

### Batch Processing

For bulk conversions:
1. Modify the conversion loop
2. Process multiple checkpoints per cycle
3. Implement parallel processing if needed

### Integration with CI/CD

The converter can be integrated into pipelines:
1. Trigger on new model releases
2. Automate deployment after conversion
3. Implement quality checks

## Development

### Extending Functionality

To enhance the converter:
1. Add support for different formats
2. Implement compression options
3. Add model validation steps
4. Enhance metrics reporting
5. Add more upload destinations
6. Implement upload retry mechanisms
7. Add model metadata management

### Contributing

When contributing:
1. Follow the existing code structure
2. Add appropriate error handling
3. Update documentation
4. Test with various model sizes

## Additional Resources

- [GGUF Format Documentation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [Templar Documentation](./README.md)
- [Model Deployment Guide](./deployment.md)