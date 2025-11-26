"""
Test Suite for Music Separator
================================

Tests the music separation functionality with sample audio files.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from music_separator import MusicSeparator


def test_music_separation():
    """
    Test music separation with a sample audio file.
    
    Instructions:
    1. Place your test music file in this directory or provide full path
    2. Update the INPUT_FILE variable below
    3. Run this script
    """
    
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================
    
    # Input: Path to your music file (mp3, wav, flac, etc.)
    INPUT_FILE = r"C:\Users\user\Desktop\mfesh_9a7eb.mp3"
    
    # Output: Directory where separated stems will be saved
    OUTPUT_DIR = r"C:\Users\user\Desktop\test_music_output"
    
    # Model: Choose separation model
    # Options: "spleeter:2stems", "spleeter:4stems", "spleeter:5stems"
    MODEL = "spleeter:4stems"
    
    # ========================================================================
    
    print("=" * 80)
    print("MUSIC SEPARATOR - TEST SUITE")
    print("=" * 80)
    print(f"\nInput File: {INPUT_FILE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print()
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print("⚠ WARNING: Input file not found!")
        print(f"Please update INPUT_FILE in test.py to point to your music file.")
        print(f"Expected: {INPUT_FILE}")
        return
    
    # Initialize separator
    separator = MusicSeparator(model=MODEL)
    
    # Perform separation
    try:
        result = separator.separate(
            input_audio=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            codec="wav"  # Output format: wav, mp3, flac, etc.
        )
        
        # Display detailed results
        print("\n" + "=" * 80)
        print("📊 DETAILED RESULTS")
        print("=" * 80)
        
        # Stem information
        print("\n🎵 Separated Stems:")
        print("-" * 80)
        for instrument, info in result['info'].items():
            print(f"\n  {instrument.upper()}:")
            print(f"    📁 File: {info['filename']}")
            print(f"    💾 Size: {info['size_mb']:.2f} MB")
            print(f"    📊 Peak Level: {info['peak_level']:.3f}")
            print(f"    📈 RMS Level: {info['rms_level']:.3f}")
            print(f"    ⏱️  Duration: {info['duration']:.2f}s")
            print(f"    📍 Path: {info['path']}")
        
        # Statistics
        stats = result['stats']
        print("\n" + "-" * 80)
        print("📈 Processing Statistics:")
        print("-" * 80)
        print(f"  ⏱️  Total Processing Time: {stats['total_time']:.2f}s ({stats['total_time']/60:.2f} min)")
        print(f"  🎼 Audio Duration: {stats['input_duration']:.2f}s ({stats['input_duration']/60:.2f} min)")
        print(f"  ⚡ Processing Speed: {stats['speed_ratio']:.2f}x realtime")
        print(f"  📥 Input File Size: {stats['input_size_mb']:.2f} MB")
        print(f"  📤 Total Output Size: {stats['output_size_mb']:.2f} MB")
        print(f"  📊 Size Expansion: {stats['output_size_mb']/stats['input_size_mb']:.2f}x")
        
        # Quality metrics
        print("\n" + "-" * 80)
        print("🎯 Quality Indicators:")
        print("-" * 80)
        for instrument, info in result['info'].items():
            quality = "High" if info['peak_level'] > 0.3 else "Medium" if info['peak_level'] > 0.1 else "Low"
            presence = "Strong" if info['rms_level'] > 0.1 else "Moderate" if info['rms_level'] > 0.05 else "Weak"
            print(f"  {instrument.upper():12s}: Quality={quality:6s} | Presence={presence}")
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Output Directory: {OUTPUT_DIR}")
        print(f"📝 Files Created: {len(result['stems'])}")
        print("\n🎧 You can now listen to the separated stems!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR DURING SEPARATION")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_files():
    """
    Test separation on multiple files.
    """
    
    # List of files to process
    files = [
        r"C:\Users\amkma\Desktop\music\song1.mp3",
        r"C:\Users\amkma\Desktop\music\song2.mp3",
    ]
    
    output_base = r"C:\Users\amkma\Desktop\music\batch_separated"
    
    separator = MusicSeparator(model="spleeter:4stems")
    
    for i, file_path in enumerate(files, 1):
        if not os.path.exists(file_path):
            print(f"Skipping {file_path} (not found)")
            continue
        
        # Create separate output directory for each file
        file_name = Path(file_path).stem
        output_dir = os.path.join(output_base, file_name)
        
        print(f"\n[{i}/{len(files)}] Processing: {file_name}")
        separator.separate(file_path, output_dir)


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("\nSelect test mode:")
    print("1. Single file separation (default)")
    print("2. Multiple files batch processing")
    
    # Run single file test by default
    test_music_separation()
    
    # Uncomment below to test batch processing:
    # test_multiple_files()
