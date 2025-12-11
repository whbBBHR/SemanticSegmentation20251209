# Project Structure

```
Semantic_Claude20251209/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   └── efficient_segmentation.py
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   └── training_pipeline.py
│   ├── inference/                # Inference and optimization
│   │   ├── __init__.py
│   │   ├── inference_optimization.py
│   │   └── pytorch_vs_tensorrt.py
│   └── utils/                    # Utility functions
│       └── __init__.py
│
├── tests/                        # Test files
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── test_m_series.py          # M-series GPU benchmarks
│
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed datasets
│
├── configs/                      # Configuration files
│   └── (training configs, model configs)
│
├── checkpoints/                  # Model checkpoints
│   └── (saved .pth/.pt files)
│
├── logs/                         # Training logs
│   └── (tensorboard, wandb logs)
│
├── scripts/                      # Utility scripts
│   └── (setup, download, preprocessing)
│
├── docs/                         # Documentation
│   └── (architecture docs, guides)
│
├── examples/                     # Example usage
│   └── (jupyter notebooks, demos)
│
├── .githooks/                    # Git hooks
│   └── pre-commit
├── .gitignore
├── README.md
├── SECURITY.md
├── requirements.txt
└── venv/                         # Virtual environment
```

## Directory Descriptions

### `src/` - Source Code
- **models/** - Neural network architectures (FastSegNet, encoders, decoders)
- **training/** - Training loops, loss functions, data augmentation
- **inference/** - Deployment, optimization (ONNX, TensorRT)
- **utils/** - Helper functions, visualization, metrics

### `tests/` - Testing
- **unit/** - Unit tests for individual components
- **integration/** - End-to-end integration tests
- **test_m_series.py** - GPU performance benchmarks

### `data/` - Datasets
- **raw/** - Original downloaded datasets
- **processed/** - Preprocessed data ready for training

### `configs/` - Configuration
- Training hyperparameters (YAML/JSON)
- Model configurations
- Dataset paths and settings

### `checkpoints/` - Saved Models
- Best models during training
- Epoch checkpoints
- ONNX/TensorRT engines

### `logs/` - Training Logs
- TensorBoard event files
- Weights & Biases logs
- Training metrics CSV

### `scripts/` - Utility Scripts
- Dataset download scripts
- Preprocessing pipelines
- Model conversion tools

### `docs/` - Documentation
- Architecture design documents
- API documentation
- Training guides

### `examples/` - Usage Examples
- Jupyter notebooks with tutorials
- Demo applications
- Sample inference code

## Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Training
python -m src.training.training_pipeline

# Inference
python -m src.inference.inference_optimization

# Testing
python -m tests.test_m_series
```
