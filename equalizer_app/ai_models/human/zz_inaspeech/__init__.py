"""
inaSpeechSegmenter Gender Detection Module
==========================================

This module provides gender detection functionality using inaSpeechSegmenter.

Main Components:
- GenderDetector: Main class for gender detection
- detect_gender_from_file: Convenience function for quick detection

Usage:
    from zz_inaspeech import GenderDetector, detect_gender_from_file
    
    # Quick usage
    results = detect_gender_from_file('audio.wav')
    
    # Advanced usage
    detector = GenderDetector(detect_gender=True)
    results = detector.detect('audio.wav')
    detector.print_results(results)

Virtual Environment: equalizer_app/.venvan/.inaspeechan_venv
"""

__version__ = "1.0.0"
__author__ = "Signal Equalizer Team"

from .gender_detector import GenderDetector, detect_gender_from_file

__all__ = ['GenderDetector', 'detect_gender_from_file']
