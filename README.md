Vietnamese Conformer CTC ASR
High-performance Vietnamese speech recognition system using Conformer encoder with CTC decoder architecture.
🚀 Project Overview
This project builds a complete ASR system with modern architecture:

Conformer Encoder: State-of-the-art architecture for speech recognition
CTC Decoder: Non-autoregressive, fast inference
SentencePiece Tokenizer: Optimized for Vietnamese with 1024 vocabulary
Advanced Training: Label smoothing, auxiliary loss, mixed precision
Auto Split: Automatic train/val split from single CSV file
Rich Augmentation: Adaptive augmentation based on audio duration

📊 Model Architecture
Encoder Architecture

ConformerEncoder: Built-from-scratch implementation with Conformer blocks

Multi-head self-attention
Depthwise separable convolution
Position-wise feed-forward networks
Macaron-style architecture


EfficientConformerEncoder: Wrapper for torchaudio.models.Conformer

Optimized performance
Compatible interface



CTC Pipeline
Audio → Mel Spectrogram → Conformer Encoder → CTC Head → Predictions
Key Components

AdvancedCTCHead: CTC projection with layer norm + dropout
AdvancedCTCDecoder: Greedy + prefix beam search decoding
CTCLossWithLabelSmoothing: CTC loss with regularization

🛠️ Installation
Dependencies
bash# Clone repository
git clone https://github.com/iamdinhthuan/EfficiencyConformer-ctc-vietnamese
cd EfficiencyConformer-ctc-vietnamese


📁 Data Structure
CSV Format
csvpath|text
./datatest/audio1.wav|transcript of the first audio file
./datatest/audio2.wav|transcript of the second audio file
Project Structure
conformer_asr/
├── metadata.csv              # Training data
├── datatest/                 # Audio files
│   ├── audio1.wav
│   └── audio2.wav
├── weights/                  # Model components
│   └── tokenizer_spe_bpe_v1024_pad/
│       └── tokenizer.model
├── config.json              # Training configuration
└── checkpoints/             # Training outputs
⚙️ Configuration
Default Configuration
pythonfrom config import ExperimentConfig

config = ExperimentConfig()
# Conformer: 256d-4h-16l
# Training: batch=16, lr=1e-4  
# Data: 95% train, 5% val
# Tokenizer: 1024 vocab SentencePiece
Custom Configuration
pythonconfig = ExperimentConfig()

# Conformer architecture
config.model.n_state = 512           # Model dimension
config.model.n_head = 8              # Attention heads
config.model.n_layer = 12            # Conformer layers
config.model.encoder_type = "conformer"  # or "efficient"
config.model.dropout = 0.1

# Training setup
config.training.batch_size = 32
config.training.learning_rate = 2e-4
config.training.max_epochs = 100
config.training.precision = "bf16-mixed"

# Data processing
config.data.metadata_file = "my_data.csv"
config.data.train_val_split = 0.9
config.data.enable_augmentation = True
config.data.min_text_len = 1
config.data.max_text_len = 60

# Save configuration
config.save("my_config.json")
🏃‍♂️ Training
Basic Training
bash# Train with default config
python run.py

# Train with custom config
python run.py --config my_config.json

# Override parameters
python run.py --batch-size 32 --learning-rate 2e-4 --max-epochs 50
Advanced Training
bash# Test setup before training
python run.py --test

# Fast development run
python run.py --fast-dev-run

# Resume from checkpoint
python run.py --resume checkpoints/ctc-step1000-wer0.1234.ckpt

# Enable profiling
python run.py --profile
🎯 Inference
Command Line
bash# Greedy decoding (fast)
python inference.py --checkpoint checkpoints/best.ckpt --audio audio.wav

# Prefix beam search (more accurate)
python inference.py --checkpoint checkpoints/best.ckpt --audio audio.wav --beam_search
Python API
pythonfrom inference import CTCInference
from config import ExperimentConfig

# Load model
config = ExperimentConfig.load("checkpoints/config.json")
inference = CTCInference("checkpoints/best.ckpt", config)

# Transcribe single file
result = inference.transcribe_single("audio.wav", use_beam_search=True)
print(f"Text: {result.transcription}")
print(f"Confidence: {result.confidence_score:.3f}")
print(f"Time: {result.processing_time:.2f}s")
🧠 Conformer Architecture Details
Conformer Block Structure
Input → FFN₁ (×0.5) → Multi-Head Attention → Convolution → FFN₂ (×0.5) → Output
Implementation Features

Macaron FFN: Split feed-forward with 0.5 scaling
Depthwise Conv: Efficient local pattern modeling
ALiBi Position: Relative position encoding (optional)
Subsampling: 4x time reduction with Conv2D

Encoder Options
ConformerEncoder (models/conformer.py)
python# Custom implementation
- Full control over architecture
- Learnable positional encoding
- Configurable conv kernel sizes
EfficientConformerEncoder (models/efficient_conformer.py)
python# TorchAudio wrapper
- Optimized performance
- Battle-tested implementation
- Same interface compatibility
📊 CTC Decoding Strategies
Greedy Decoding
python# O(T) time complexity
# Fastest inference
decoded = ctc_decoder.greedy_decode(log_probs, lengths)
Prefix Beam Search
python# O(T × B × V) complexity
# Higher accuracy
decoded = ctc_decoder.prefix_beam_search(
    log_probs, lengths, 
    beam_size=5, alpha=0.3
)
📈 Training Features
Multi-task Learning

Main CTC Loss: From final encoder output
Auxiliary Loss: From intermediate layers (25%, 50%, 75%)
Total Loss: main_loss + aux_weight × aux_loss

Optimization Strategy

OneCycleLR: Warmup → peak → cosine decay
Mixed Precision: bf16 to save memory
Gradient Clipping: Stability
Label Smoothing: Regularization

Data Augmentation
python# Adaptive based on audio duration
Short (< 3s):    Basic gain + noise
Medium (3-8s):   + Time stretch + background noise  
Long (> 8s):     + Stronger augmentation
Noisy env:       MP3 compression + low SNR
🔧 Advanced Configuration
Model Scaling
python# Small model (fast)
config.model.n_state = 256
config.model.n_layer = 12
config.model.n_head = 4

# Large model (accurate)
config.model.n_state = 512
config.model.n_layer = 24
config.model.n_head = 8

# Efficient variant
config.model.encoder_type = "efficient"
Training Optimization
python# Memory efficient
config.training.batch_size = 8
config.training.precision = "bf16-mixed"
config.training.accumulate_grad_batches = 4

# Speed optimized
config.training.num_workers = 0  # Single-threaded
config.training.num_sanity_val_steps = 0
📋 Monitoring & Debugging
TensorBoard Metrics
bashtensorboard --logdir checkpoints
Key Metrics:

train_loss / val_loss_epoch: CTC loss
train_wer / val_wer_epoch: Word Error Rate
learning_rate: Current learning rate
step_time: Training speed
aux_loss: Auxiliary CTC loss

Common Issues
Memory Problems
pythonconfig.training.batch_size = 8          # Reduce batch size
config.training.accumulate_grad_batches = 4  # Gradient accumulation
Slow Training
pythonconfig.model.encoder_type = "efficient"  # Use TorchAudio
config.training.precision = "bf16-mixed" # Mixed precision
config.training.num_workers = 0          # Single worker
NaN Loss
pythonconfig.training.learning_rate = 5e-5     # Lower learning rate
config.training.gradient_clip_val = 0.5  # Stronger clipping
🎛️ Complete Configuration Example
pythonconfig = ExperimentConfig(
    name="vietnamese_conformer_large",
    
    # Audio processing
    audio=AudioConfig(
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160
    ),
    
    # Conformer architecture
    model=ModelConfig(
        n_state=512,
        n_head=8, 
        n_layer=18,
        encoder_type="conformer",
        vocab_size=1024,
        dropout=0.1,
        label_smoothing=0.1
    ),
    
    # Training setup
    training=TrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        max_epochs=100,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        aux_loss_weight=0.2
    ),
    
    # Data configuration
    data=DataConfig(
        metadata_file="metadata.csv",
        train_val_split=0.95,
        enable_augmentation=True,
        min_text_len=1,
        max_text_len=60
    )
)
🔄 Complete Workflow

Prepare Data
bash# Create metadata.csv with path|text format
# Ensure audio files exist

Configure Model
python# Customize config.json for your needs
# Test with small config first

Test Setup
bashpython run.py --test

Start Training
bashpython run.py --config config.json

Monitor Progress
bashtensorboard --logdir checkpoints

Run Inference
bashpython inference.py --checkpoint checkpoints/best.ckpt --audio test.wav


🤝 Development
Architecture Overview
models/
├── conformer.py              # Native Conformer implementation
├── efficient_conformer.py    # TorchAudio wrapper
├── advanced_ctc.py          # CTC components
└── encoder.py               # Legacy reference

utils/
├── dataset.py               # Data processing
└── scheduler.py             # LR scheduling

config.py                    # Configuration system
train.py                     # Lightning module
run.py                       # Training script
inference.py                 # Inference engine
Extending the System

New Encoder: Implement interface in models/
New Augmentation: Extend AdvancedAudioAugmentation
New Tokenizer: Update config + dataset code
New Loss: Modify CTCLossWithLabelSmoothing

📄 License
MIT License
🙏 Credits

Conformer Architecture: "Conformer: Convolution-augmented Transformer for Speech Recognition"
PyTorch Lightning: Training framework
TorchAudio: Efficient Conformer implementation
SentencePiece: Subword tokenization


⭐ Star the repo if useful! ⭐
