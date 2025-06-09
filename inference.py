import torch
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
from loguru import logger
import librosa
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import time
import concurrent.futures
from dataclasses import dataclass
import json
from tqdm import tqdm

# Import model components
from models.encoder import AudioEncoder
from models.advanced_ctc import AdvancedCTCHead, AdvancedCTCDecoder
from config import ExperimentConfig, get_config


@dataclass
class InferenceResult:
    """Container for inference results"""
    file_path: str
    transcription: str
    confidence_score: float
    processing_time: float
    method: str  # 'greedy' or 'beam_search'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'transcription': self.transcription,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'method': self.method
        }


class CTCInference:
    """Advanced CTC inference with optimization strategies"""
    
    def __init__(self, checkpoint_path: str, config: Optional[ExperimentConfig] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or get_config()
        logger.info(f"🚀 Initializing CTC inference on {self.device}")
        
        self._load_model(checkpoint_path)
        self._init_tokenizer()
        self._init_decoder()
        
        logger.info("✅ Inference engine ready!")
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        logger.info(f"📦 Loading model from {checkpoint_path}")
        
        # Initialize model components
        self.encoder = AudioEncoder(
            n_mels=self.config.audio.n_mels,
            n_state=self.config.model.n_state,
            n_head=self.config.model.n_head,
            n_layer=self.config.model.n_layer,
            att_context_size=self.config.model.attention_context_size
        )
        
        self.ctc_head = AdvancedCTCHead(
            input_dim=self.config.model.n_state,
            vocab_size=self.config.model.vocab_size,
            dropout=0.0  # No dropout during inference
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Separate encoder and CTC head weights
        encoder_weights = {}
        ctc_weights = {}
        
        for key, value in state_dict.items():
            if 'alibi' in key:  # Skip ALiBi weights
                continue
            elif key.startswith('encoder.'):
                encoder_weights[key.replace('encoder.', '')] = value
            elif key.startswith('ctc_head.'):
                ctc_weights[key.replace('ctc_head.', '')] = value
        
        # Load weights
        self.encoder.load_state_dict(encoder_weights, strict=False)
        self.ctc_head.load_state_dict(ctc_weights, strict=False)
        
        # Move to device and set to eval mode
        self.encoder = self.encoder.to(self.device).eval()
        self.ctc_head = self.ctc_head.to(self.device).eval()
        
    def _init_tokenizer(self):
        """Initialize SentencePiece tokenizer"""
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.config.model.tokenizer_model_path)
        
    def _init_decoder(self):
        """Initialize CTC decoder"""
        self.decoder = AdvancedCTCDecoder(self.config.model.vocab_size, self.config.model.rnnt_blank)
    
    def log_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log mel spectrogram"""
        window = torch.hann_window(self.config.audio.n_fft).to(audio.device)
        stft = torch.stft(audio, self.config.audio.n_fft, self.config.audio.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        
        # Use librosa mel filters
        mel_basis = librosa.filters.mel(sr=self.config.audio.sample_rate, n_fft=self.config.audio.n_fft, n_mels=self.config.audio.n_mels)
        mel_basis = torch.from_numpy(mel_basis).to(audio.device)
        
        mel_spec = torch.matmul(mel_basis, magnitudes)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = (log_spec + 4.0) / 4.0
        
        return log_spec
    
    def transcribe_single(self, audio_path: str, use_beam_search: bool = False) -> InferenceResult:
        """Transcribe single audio file"""
        start_time = time.time()
        
        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=self.config.audio.sample_rate)
            audio_tensor = torch.from_numpy(audio).to(self.device)
            
            with torch.no_grad():
                # Compute features
                mels = self.log_mel_spectrogram(audio_tensor)
                x = mels.unsqueeze(0)  # Add batch dimension
                x_len = torch.tensor([x.shape[2]]).to(self.device)
                
                # Forward pass
                enc_out, enc_len = self.encoder(x, x_len)
                logits = self.ctc_head(enc_out)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Decode
                if use_beam_search:
                    decoded_sequences = self.decoder.prefix_beam_search(log_probs, enc_len)
                    method = "beam_search"
                else:
                    decoded_sequences = self.decoder.greedy_decode(log_probs, enc_len)
                    method = "greedy"
                
                # Get transcription
                transcription = self.tokenizer.decode(decoded_sequences[0]) if decoded_sequences[0] else ""
                confidence = 0.8  # Placeholder confidence score
                
        except Exception as e:
            logger.error(f"❌ Error processing {audio_path}: {e}")
            transcription = ""
            confidence = 0.0
            method = "error"
        
        processing_time = time.time() - start_time
        
        return InferenceResult(
            file_path=audio_path,
            transcription=transcription,
            confidence_score=confidence,
            processing_time=processing_time,
            method=method
        )


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CTC ASR Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--beam_search", action="store_true", help="Use beam search")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Initialize inference
    inference = CTCInference(args.checkpoint, config, args.device)
    
    # Transcribe
    result = inference.transcribe_single(args.audio, args.beam_search)
    
    print(f"🎯 Transcription: {result.transcription}")
    print(f"⏱️ Time: {result.processing_time:.2f}s")
    print(f"📈 Confidence: {result.confidence_score:.3f}")


if __name__ == "__main__":
    main() 