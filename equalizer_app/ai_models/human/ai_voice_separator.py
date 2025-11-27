import os
import json
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from functools import partial
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

import torch
import torchaudio
import numpy as np
import librosa

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
    torchaudio.get_audio_backend = lambda: 'soundfile'

try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass


@dataclass
class SpeakerProfile:
    id: int
    audio_path: str
    audio_tensor: torch.Tensor
    sample_rate: int
    duration: float
    characteristics: Dict[str, any]
    
    def __init__(self, speaker_id: int, audio_path: str, audio_tensor: torch.Tensor, sr: int):
        self.id = speaker_id
        self.audio_path = audio_path
        self.audio_tensor = audio_tensor
        self.sample_rate = sr
        self.duration = audio_tensor.shape[-1] / sr
        self.characteristics = {
            'gender': 'unknown',
            'gender_confidence': 0.0,
            'age': 'unknown',
            'age_confidence': 0.0,
            'language': 'unknown',
            'language_confidence': 0.0
        }
    
    def get_filename(self) -> str:
        lang = self.characteristics['language']
        lang_clean = lang.replace(':', '_').replace(' ', '_').replace('/', '_')
        return f"speaker_{self.id:02d}_{self.characteristics['gender']}_" \
               f"{self.characteristics['age']}_{lang_clean}.wav"


class AIVoiceSeparator:
    def __init__(self):
        self.models_dir = Path(__file__).parent / "ai_models_cache"
        self.models_dir.mkdir(exist_ok=True)
        
        # Disable symlinks for Windows compatibility
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["SPEECHBRAIN_CACHE"] = str(self.models_dir)
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        
        self.gender_detector_dir = Path(__file__).parent / "zz_inaspeech"
        self.gender_venv_path = Path(__file__).parent.parent.parent / ".venvan" / ".inaspeechan_venv"
        
        self.separator = None
        self.age_model = None
        self.age_processor = None
        self.language_classifier = None
        
        self._load_models()
    
    def _load_models(self):
        # Load MossFormer2 separator
        try:
            # Add asteroid model directory to path
            asteroid_dir = Path(__file__).parent.parent.parent.parent / "asteroid"
            model_dir = asteroid_dir / "egs" / "wsj0-mix-var" / "Multi-Decoder-DPRNN"
            
            if not model_dir.exists():
                raise RuntimeError(f"MultiDecoder DPRNN model directory not found: {model_dir}")
            
            sys.path.insert(0, str(model_dir))
            
            # Patch torch.load to allow pickle
            original_load = torch.load
            torch.load = partial(original_load, weights_only=False)
            
            print("  Loading MultiDecoder DPRNN...")
            from model import MultiDecoderDPRNN
            self.separator = MultiDecoderDPRNN.from_pretrained("JunzheJosephZhu/MultiDecoderDPRNN").eval()
            
            # Restore torch.load
            torch.load = original_load
            
            print(f"  ✓ MultiDecoder DPRNN loaded (variable sources)")
            self.separator_type = "multidecoder"
            
        except Exception as e:
            raise RuntimeError(f"Cannot load MultiDecoder DPRNN: {e}")
        
        if not self.gender_venv_path.exists():
            raise RuntimeError(f"Gender detector venv not found: {self.gender_venv_path}")
        if not (self.gender_detector_dir / "gender_detector.py").exists():
            raise RuntimeError("Gender detector script not found")
        
        try:
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
            self.age_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "bookbot/wav2vec2-adult-child-cls",
                cache_dir=str(self.models_dir / "wav2vec2-adult-child-cls")
            )
            self.age_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "bookbot/wav2vec2-adult-child-cls",
                cache_dir=str(self.models_dir / "wav2vec2-adult-child-cls")
            )
            self.age_model.eval()
        except Exception as e:
            raise RuntimeError(f"Cannot load age classifier: {e}")
        
        # try:
        #     from speechbrain.pretrained import EncoderClassifier
        #     self.language_classifier = EncoderClassifier.from_hparams(
        #         source="speechbrain/lang-id-voxlingua107-ecapa",
        #         savedir=str(self.models_dir / "lang-id-voxlingua107-ecapa"),
        #         run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        #     )
        # except Exception as e:
        #     raise RuntimeError(f"Cannot load language ID model: {e}")
    
    def run_pipeline(self, input_audio: str, output_dir: str):
        temp_dir = tempfile.mkdtemp(prefix="voice_sep_")
        
        try:
            speakers = self._separate_sources(input_audio, temp_dir)
            if not speakers:
                return
            
            for speaker in speakers:
                # gender, gender_conf = self._classify_gender(speaker.audio_path)
                # speaker.characteristics['gender'] = gender
               # speaker.characteristics['gender_confidence'] = gender_conf
                
                age, age_conf = self._classify_age(speaker.audio_tensor, speaker.sample_rate)
                speaker.characteristics['age'] = age
                speaker.characteristics['age_confidence'] = age_conf
                
                lang, lang_conf = self._classify_language(speaker.audio_tensor, speaker.sample_rate)
                speaker.characteristics['language'] = lang
                speaker.characteristics['language_confidence'] = lang_conf
            
            self._save_outputs(speakers, output_dir)
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _separate_sources(self, input_audio: str, temp_dir: str) -> List[SpeakerProfile]:
        """
        Voice separation without recursion.
        
        Args:
            input_audio: Path to input audio file
            temp_dir: Temporary directory
            
        Returns:
            List of all detected speakers
        """
        print(f"Input: {input_audio}")
        
        # Single-pass separation
        all_speakers = self._perform_separation(input_audio, temp_dir, depth=0)
        
        if not all_speakers:
            print("  ✗ No valid sources detected")
            return []
        
        # Renumber speakers sequentially
        for idx, speaker in enumerate(all_speakers, 1):
            speaker.id = idx
        
        print(f"\n  ✓ Final: {len(all_speakers)} speaker(s) detected")
        return all_speakers
    
    def _perform_separation(self, input_audio: str, temp_dir: str, depth: int) -> List[SpeakerProfile]:
        """
        Core separation engine with quality validation.
        
        Args:
            input_audio: Audio file path
            temp_dir: Temporary directory
            depth: 0=initial, 1=re-separation
            
        Returns:
            List of valid speaker sources
        """
        indent = "    " if depth > 0 else "  "
        
        # ====================================================================
        # Audio Loading
        # ====================================================================
        import soundfile as sf
        
        try:
            y, sr = sf.read(input_audio)
        except Exception as e:
            print(f"{indent}✗ Audio load failed: {e}")
            return []
        
        # Convert to tensor and ensure mono
        if len(y.shape) == 1:
            audio = torch.from_numpy(y).float()  # (samples,)
        else:
            # Stereo to mono: average channels
            audio = torch.from_numpy(y).float().mean(dim=1)  # (samples,)
        
        if depth == 0:
            print(f"{indent}Loaded: {sr}Hz, {audio.shape[0]} samples ({audio.shape[0]/sr:.2f}s)")
        
        # ====================================================================
        # Preprocessing
        # ====================================================================
        # Normalize to prevent clipping
        max_val = audio.abs().max()
        if max_val > 1e-8:
            audio = audio / (max_val + 1e-8)
        else:
            print(f"{indent}✗ Audio too quiet (max: {max_val:.2e})")
            return []
        
        # Resample to 8kHz (model requirement)
        target_sr = 8000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio.unsqueeze(0)).squeeze(0)  # Add/remove channel for resampling
            sr = target_sr
        
        # Add batch dimension: (samples,) -> (1, samples)
        mixture = audio.unsqueeze(0)
        
        # ====================================================================
        # Separation
        # ====================================================================
        try:
            with torch.no_grad():
                # MultiDecoder DPRNN expects (batch, samples)
                sources_est = self.separator.separate(mixture).cpu()
                # sources_est is a list of tensors, each (samples,)
                separated = torch.stack([s for s in sources_est], dim=0)  # (n_sources, samples)
                    
        except Exception as e:
            print(f"{indent}✗ Separation failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        num_sources = separated.shape[0]
        print(f"{indent}→ {num_sources} source(s)")
        
        # ====================================================================
        # Quality Validation & Saving
        # ====================================================================
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        
        speakers = []
        quality_thresholds = {
            'min_amplitude': 0.05,
            'min_energy': 0.0005,
            'min_duration': 1.0  # seconds
        }
        
        for i in range(num_sources):
            source = separated[i]
            
            # Quality metrics
            max_amp = source.abs().max().item()
            energy = (source ** 2).mean().item()
            duration = source.shape[0] / sr
            
            # Validation
            if max_amp < quality_thresholds['min_amplitude']:
                if depth == 0:
                    print(f"{indent}  Source {i+1}: SKIP (quiet: {max_amp:.4f})")
                continue
            
            if energy < quality_thresholds['min_energy']:
                if depth == 0:
                    print(f"{indent}  Source {i+1}: SKIP (low energy: {energy:.6f})")
                continue
            
            if duration < quality_thresholds['min_duration']:
                if depth == 0:
                    print(f"{indent}  Source {i+1}: SKIP (too short: {duration:.2f}s)")
                continue
            
            # Normalize to 95% to prevent clipping
            source = source / (max_amp + 1e-8) * 0.95
            
            # Save with unique filename
            temp_file = temp_path / f"sep_d{depth}_s{i+1}_p{os.getpid()}.wav"
            
            try:
                sf.write(str(temp_file), source.numpy(), sr)
            except Exception as e:
                print(f"{indent}  Source {i+1}: SAVE FAILED ({e})")
                continue
            
            # Create speaker profile
            speaker = SpeakerProfile(
                speaker_id=i+1,
                audio_path=str(temp_file),
                audio_tensor=source,
                sr=sr
            )
            speakers.append(speaker)
            
            if depth == 0:
                print(f"{indent}  Source {i+1}: OK (amp:{max_amp:.3f}, energy:{energy:.6f}, {duration:.2f}s)")
        
        return speakers
    
#     def _classify_gender(self, audio_path: str) -> Tuple[str, float]:
#         """
#         Stage 3: Classify gender using inaSpeechSegmenter (separate venv).
#
#         Args:
#             audio_path: Path to audio file
#
#         Returns:
#             Tuple of (gender, confidence)
#         """
#         print(f"    [Gender] Analyzing with inaSpeechSegmenter...")
#
#         try:
#             # Build Python executable path in the gender detector venv
#             python_exe = self.gender_venv_path / "Scripts" / "python.exe"
#
#             if not python_exe.exists():
#                 print(f"    [Gender] ✗ Python not found: {python_exe}")
#                 return 'unknown', 0.0
#
#             # Create a simple script to run gender detection
#             script_code = f"""
# import sys
# sys.path.insert(0, r"{self.gender_detector_dir}")
# from gender_detector import GenderDetector
#
# detector = GenderDetector(detect_gender=True)
# results = detector.detect(r"{audio_path}")
# gender = results['dominant_gender']
# male_pct = results['percentages']['male']
# female_pct = results['percentages']['female']
#
# # Output format: gender|confidence
# if gender == 'male':
#     print(f"male|{{male_pct}}")
# elif gender == 'female':
#     print(f"female|{{female_pct}}")
# else:
#     print("unknown|0.0")
# """
#
#             # Run the script in the separate venv
#             result = subprocess.run(
#                 [str(python_exe), "-c", script_code],
#                 capture_output=True,
#                 text=True,
#                 timeout=30
#             )
#
#             if result.returncode != 0:
#                 print(f"    [Gender] ✗ Error: {result.stderr}")
#                 raise RuntimeError(f"Gender classification failed: {result.stderr}")
#
#             # Parse output: "gender|confidence"
#             output = result.stdout.strip().split('\n')[-1]  # Last line
#             parts = output.split('|')
#
#             if len(parts) == 2:
#                 gender = parts[0]
#                 confidence = float(parts[1])
#                 print(f"    [Gender] {gender.upper()} (confidence: {confidence:.1f}%)")
#                 return gender, confidence
#             else:
#                 print(f"    [Gender] ✗ Unexpected output: {output}")
#                 raise RuntimeError(f"Gender classification output parse error: {output}")
#
#         except subprocess.TimeoutExpired:
#             print(f"    [Gender] ✗ Timeout (>30s)")
#             raise RuntimeError("Gender classification timeout")
#         except Exception as e:
#             print(f"    [Gender] ✗ Error: {e}")
#             raise
    
    def _classify_age(self, audio_tensor: torch.Tensor, sr: int) -> Tuple[str, float]:
        """
        Stage 4: Classify age (adult/child) using wav2vec2 transformer model.
        
        Args:
            audio_tensor: Audio waveform tensor
            sr: Sample rate
            
        Returns:
            Tuple of (age_category, confidence)
        """
        print(f"    [Age] Analyzing with wav2vec2-adult-child-cls...")
        
        # Convert tensor to numpy
        y = audio_tensor.numpy()
        
        # Resample to 16kHz (required by wav2vec2)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Prepare input for the model
        inputs = self.age_processor(
            y,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        
        # Run inference
        with torch.no_grad():
            logits = self.age_model(**inputs).logits
        
        # Get prediction
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        confidence = probabilities[predicted_class_id].item() * 100
        
        # Model outputs: 0 = adult, 1 = child
        age = 'adult' if predicted_class_id == 0 else 'child'
        
        print(f"    [Age] {age.upper()} (class: {predicted_class_id}, confidence: {confidence:.1f}%)")
        return age, confidence
    
    def _classify_language(self, audio_tensor: torch.Tensor, sr: int) -> Tuple[str, float]:
        """
        Stage 5: Identify language using VoxLingua107.
        
        VoxLingua107 output format:
        - predictions[0]: log probabilities
        - predictions[1]: log probabilities (tensor)
        - predictions[2]: embeddings
        - predictions[3]: predicted language codes (list)
        
        Args:
            audio_tensor: Audio waveform tensor
            sr: Sample rate
            
        Returns:
            Tuple of (language, confidence)
        """
        print(f"    [Language] Analyzing with VoxLingua107...")
        
        # Ensure audio is 1D tensor
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor.squeeze()
        
        # Resample to 16kHz if needed (VoxLingua107 requirement)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        
        # Add batch dimension for model input
        audio_batch = audio_tensor.unsqueeze(0)
        
        # Run language classification
        with torch.no_grad():
            predictions = self.language_classifier.classify_batch(audio_batch)
        
        # Extract results from predictions tuple
        # predictions[3] contains the predicted language codes
        # predictions[1] contains log probabilities
        try:
            # Get the predicted language ID (first element of predictions[3])
            language_id = predictions[3][0]
            
            # Get confidence from log probabilities
            # predictions[1] is a tensor of log probabilities
            log_probs = predictions[1]
            
            # Convert log probabilities to probabilities and get max
            probs = torch.exp(log_probs)
            confidence = probs.max().item() * 100
            
            print(f"    [Language] {language_id.upper()} (confidence: {confidence:.1f}%)")
            return language_id, confidence
            
        except Exception as e:
            print(f"    [Language] ✗ Error extracting predictions: {e}")
            print(f"    [Language] Predictions structure: {[type(p) for p in predictions]}")
            raise RuntimeError(f"Language classification failed: {e}")
    
    def _save_outputs(self, speakers: List[SpeakerProfile], output_dir: str):
        """
        Stage 6: Save labeled audio files and metadata.
        
        Args:
            speakers: List of classified speakers
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'total_speakers': len(speakers),
            'speakers': []
        }
        
        for speaker in speakers:
            # Generate filename
            filename = speaker.get_filename()
            output_file = output_path / filename
            
            # Save audio file
            # Resample back to 16kHz for output
            if speaker.sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(speaker.sample_rate, 16000)
                audio_output = resampler(speaker.audio_tensor.unsqueeze(0))
            else:
                audio_output = speaker.audio_tensor.unsqueeze(0)
            
            # Use soundfile for saving
            import soundfile as sf
            audio_np = audio_output.squeeze(0).numpy()
            sf.write(str(output_file), audio_np, 16000)
            print(f"  ✓ {filename}")
            
            # Add to metadata
            speaker_meta = {
                'id': speaker.id,
                'filename': filename,
                'duration': speaker.duration,
                'characteristics': speaker.characteristics
            }
            metadata['speakers'].append(speaker_meta)
        
        # Save metadata JSON
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✓ metadata.json")


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    # Example usage
    separator = AIVoiceSeparator()
    
    # Test with sample audio
    input_file = "test_audio.wav"
    output_dir = "separated_output"
    
    separator.run_pipeline(input_file, output_dir)