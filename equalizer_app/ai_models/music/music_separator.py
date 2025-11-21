import os
from pathlib import Path
import warnings
import time
import json
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter


class MusicSeparator:
    def __init__(self, model: str = "spleeter:4stems"):
        print(f"Initializing MusicSeparator with model: {model}")
        self.model_name = model
        self.audio_loader = AudioAdapter.default()
        
        models_dir = Path(__file__).parent / "pretrained_models"
        models_dir.mkdir(exist_ok=True)
        
        print(f"Loading Spleeter model...")
        self.separator = Separator(model)
        print("Model loaded successfully")
        
        default_cache = Path.home() / ".cache" / "spleeter"
        if default_cache.exists():
            print(f"Note: Spleeter models downloaded to: {default_cache}")
            print(f"To use custom location, manually copy models to: {models_dir}")
    
    def separate(self, input_audio: str, output_dir: str, codec: str = "wav") -> dict:
        print(f"Starting separation: {input_audio}")
        input_path = Path(input_audio)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_audio}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        print("Loading audio...")
        waveform, sample_rate = self.audio_loader.load(str(input_path), sample_rate=44100)
        load_time = time.time() - start_time
        duration = waveform.shape[0] / sample_rate
        print(f"Audio loaded: {duration:.2f}s, {sample_rate}Hz")
        
        print("Separating sources...")
        sep_start = time.time()
        prediction = self.separator.separate(waveform)
        sep_time = time.time() - sep_start
        print(f"Separation complete in {sep_time:.2f}s")
        
        print("Saving stems...")
        stem_paths = {}
        stem_info = {}
        
        for instrument, audio_data in prediction.items():
            stem_filename = f"{input_path.stem}_{instrument}.{codec}"
            stem_path = output_path / stem_filename
            
            rms = np.sqrt(np.mean(audio_data ** 2))
            peak = np.max(np.abs(audio_data))
            
            self.audio_loader.save(str(stem_path), audio_data, sample_rate, codec=codec)
            print(f"  Saved: {instrument} -> {stem_filename}")
            
            saved_size_mb = stem_path.stat().st_size / (1024 * 1024)
            
            stem_paths[instrument] = str(stem_path)
            stem_info[instrument] = {
                'filename': stem_filename,
                'path': str(stem_path),
                'size_mb': saved_size_mb,
                'rms_level': float(rms),
                'peak_level': float(peak),
                'duration': duration
            }
        
        total_time = time.time() - start_time
        
        metadata = {
            'input': {
                'filename': input_path.name,
                'path': str(input_path),
                'duration': duration,
                'sample_rate': sample_rate
            },
            'output': {
                'directory': str(output_path),
                'codec': codec,
                'stems_count': len(stem_paths)
            },
            'stems': stem_info,
            'processing': {
                'model': self.model_name,
                'load_time': load_time,
                'separation_time': sep_time,
                'total_time': total_time
            }
        }
        
        metadata_path = output_path / f"{input_path.stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - start_time
        
        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        output_size_mb = sum(info['size_mb'] for info in stem_info.values())
        speed_ratio = duration / total_time if total_time > 0 else 0
        
        print(f"Complete! Total time: {total_time:.2f}s")
        print(f"Output: {output_dir}")
        
        return {
            'stems': stem_paths,
            'info': stem_info,
            'stats': {
                'input_duration': duration,
                'total_time': total_time,
                'separation_time': sep_time,
                'speed_ratio': speed_ratio,
                'input_size_mb': input_size_mb,
                'output_size_mb': output_size_mb
            },
            'metadata_path': str(metadata_path)
        }
