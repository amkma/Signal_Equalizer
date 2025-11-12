"""
Gender Detection using inaSpeechSegmenter
==========================================
This module uses inaSpeechSegmenter to detect gender from audio input.
It segments audio and identifies male/female speech portions.

Virtual Environment: equalizer_app/.venvan/.inaspeechan_venv
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from inaSpeechSegmenter import Segmenter
    from inaSpeechSegmenter.features import media2feats
except ImportError as e:
    print(f"Error: inaSpeechSegmenter not found. Please activate the virtual environment:")
    print(f"  equalizer_app/.venvan/.inaspeechan_venv/Scripts/activate")
    print(f"  pip install inaSpeechSegmenter")
    sys.exit(1)


class GenderDetector:
    """
    Gender detection using inaSpeechSegmenter.
    
    Detects:
    - male: Male speech
    - female: Female speech
    - noEnergy: Silence/no energy
    - noise: Background noise
    - music: Music segments
    """
    
    def __init__(self, detect_gender=True, vad_engine='sm', batch_size=32):
        """
        Initialize the gender detector.
        
        Args:
            detect_gender (bool): Enable gender detection (male/female)
            vad_engine (str): Voice Activity Detection engine
                - 'sm': SpeechMusic segmentation (default)
                - 'smn': SpeechMusicNoise segmentation
            batch_size (int): Processing batch size
        """
        print("Initializing inaSpeechSegmenter...")
        self.detect_gender = detect_gender
        self.vad_engine = vad_engine
        self.batch_size = batch_size
        
        # Initialize the segmenter
        self.segmenter = Segmenter(
            vad_engine=vad_engine,
            detect_gender=detect_gender,
            batch_size=batch_size
        )
        print("✓ inaSpeechSegmenter initialized successfully")
    
    def detect(self, audio_path):
        """
        Detect gender and speech segments from audio file.
        
        Args:
            audio_path (str): Path to audio file (wav, mp3, etc.)
        
        Returns:
            dict: Detection results with segments and statistics
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"\nProcessing: {os.path.basename(audio_path)}")
        print("-" * 60)
        
        # Perform segmentation
        segments = self.segmenter(audio_path)
        
        # Convert to list for processing
        segment_list = list(segments)
        
        # Analyze results
        results = self._analyze_segments(segment_list)
        results['audio_path'] = audio_path
        results['segments'] = segment_list
        
        return results
    
    def _analyze_segments(self, segments):
        """
        Analyze segmentation results and compute statistics.
        
        Args:
            segments (list): List of (label, start_time, end_time) tuples
        
        Returns:
            dict: Analysis results with durations and percentages
        """
        stats = {
            'male': 0.0,
            'female': 0.0,
            'noEnergy': 0.0,
            'noise': 0.0,
            'music': 0.0
        }
        
        total_duration = 0.0
        
        # Calculate durations for each segment type
        for label, start, end in segments:
            duration = end - start
            if label in stats:
                stats[label] += duration
            total_duration += duration
        
        # Calculate percentages
        percentages = {}
        for label, duration in stats.items():
            if total_duration > 0:
                percentages[label] = (duration / total_duration) * 100
            else:
                percentages[label] = 0.0
        
        # Determine dominant gender
        dominant_gender = None
        if stats['male'] > stats['female']:
            dominant_gender = 'male'
        elif stats['female'] > stats['male']:
            dominant_gender = 'female'
        elif stats['male'] > 0 or stats['female'] > 0:
            dominant_gender = 'both'
        
        # Calculate speech-to-total ratio
        speech_duration = stats['male'] + stats['female']
        speech_ratio = (speech_duration / total_duration * 100) if total_duration > 0 else 0.0
        
        return {
            'durations': stats,
            'percentages': percentages,
            'total_duration': total_duration,
            'speech_duration': speech_duration,
            'speech_ratio': speech_ratio,
            'dominant_gender': dominant_gender,
            'num_segments': len(segments)
        }
    
    def print_results(self, results):
        """
        Print detection results in a formatted way.
        
        Args:
            results (dict): Detection results from detect()
        """
        print("\n" + "=" * 60)
        print("GENDER DETECTION RESULTS")
        print("=" * 60)
        
        print(f"\nAudio File: {os.path.basename(results['audio_path'])}")
        print(f"Total Duration: {results['total_duration']:.2f} seconds")
        print(f"Number of Segments: {results['num_segments']}")
        
        print("\n" + "-" * 60)
        print("SEGMENT ANALYSIS")
        print("-" * 60)
        
        durations = results['durations']
        percentages = results['percentages']
        
        # Print each category
        categories = [
            ('male', '🔵 Male Speech'),
            ('female', '🔴 Female Speech'),
            ('music', '🎵 Music'),
            ('noise', '📢 Noise'),
            ('noEnergy', '🔇 Silence')
        ]
        
        for key, label in categories:
            duration = durations[key]
            percentage = percentages[key]
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            print(f"{label:20s}: {duration:6.2f}s ({percentage:5.1f}%) {bar}")
        
        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        
        print(f"Speech Duration: {results['speech_duration']:.2f}s ({results['speech_ratio']:.1f}%)")
        
        if results['dominant_gender']:
            gender_emoji = '🔵' if results['dominant_gender'] == 'male' else '🔴' if results['dominant_gender'] == 'female' else '🟣'
            print(f"Dominant Gender: {gender_emoji} {results['dominant_gender'].upper()}")
        else:
            print("Dominant Gender: No speech detected")
        
        print("\n" + "-" * 60)
        print("DETAILED SEGMENTS")
        print("-" * 60)
        print(f"{'Label':<15} {'Start (s)':>10} {'End (s)':>10} {'Duration (s)':>12}")
        print("-" * 60)
        
        for label, start, end in results['segments']:
            duration = end - start
            emoji = {
                'male': '🔵',
                'female': '🔴',
                'music': '🎵',
                'noise': '📢',
                'noEnergy': '🔇'
            }.get(label, '  ')
            print(f"{emoji} {label:<13} {start:>10.2f} {end:>10.2f} {duration:>12.2f}")
        
        print("=" * 60 + "\n")
    
    def get_gender_summary(self, results):
        """
        Get a simple gender summary string.
        
        Args:
            results (dict): Detection results from detect()
        
        Returns:
            str: Simple gender summary
        """
        dominant = results['dominant_gender']
        male_pct = results['percentages']['male']
        female_pct = results['percentages']['female']
        
        if dominant == 'male':
            return f"Male voice detected ({male_pct:.1f}% of audio)"
        elif dominant == 'female':
            return f"Female voice detected ({female_pct:.1f}% of audio)"
        elif dominant == 'both':
            return f"Both genders detected (Male: {male_pct:.1f}%, Female: {female_pct:.1f}%)"
        else:
            return "No speech detected"


def detect_gender_from_file(audio_path, verbose=True):
    """
    Convenience function to detect gender from an audio file.
    
    Args:
        audio_path (str): Path to audio file
        verbose (bool): Print detailed results
    
    Returns:
        dict: Detection results
    """
    detector = GenderDetector(detect_gender=True)
    results = detector.detect(audio_path)
    
    if verbose:
        detector.print_results(results)
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Gender Detector - inaSpeechSegmenter")
    print("=" * 60)
    print("\nUsage:")
    print("  from gender_detector import GenderDetector, detect_gender_from_file")
    print("\nExample:")
    print("  detector = GenderDetector()")
    print("  results = detector.detect('audio.wav')")
    print("  detector.print_results(results)")
    print("\nOr use the convenience function:")
    print("  results = detect_gender_from_file('audio.wav')")
    print("=" * 60)
