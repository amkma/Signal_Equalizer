"""
Quick Test - Gender Detection
==============================
Simple test to verify inaSpeechSegmenter is working.

This will test with any audio file you provide.
"""

import os
import sys
from pathlib import Path

# Check if running in correct virtual environment
venv_path = Path(__file__).parent.parent.parent.parent / '.venvan' / '.inaspeechan_venv'
expected_venv = str(venv_path.resolve())

print("=" * 70)
print(" QUICK GENDER DETECTION TEST")
print("=" * 70)
print(f"\nExpected Virtual Environment:")
print(f"  {expected_venv}")

# Check if inaSpeechSegmenter is available
try:
    import inaSpeechSegmenter
    print(f"\n✓ inaSpeechSegmenter is installed (version: {inaSpeechSegmenter.__version__})")
except ImportError:
    print("\n❌ inaSpeechSegmenter NOT found!")
    print("\nPlease activate the virtual environment:")
    print("  cd equalizer_app\\.venvan\\.inaspeechan_venv")
    print("  .\\Scripts\\Activate.ps1")
    print("\nThen install dependencies:")
    print("  pip install inaSpeechSegmenter==0.7.6")
    sys.exit(1)

# Import our module
from gender_detector import detect_gender_from_file


def quick_test(audio_file):
    """Run a quick test on an audio file."""
    
    if not os.path.exists(audio_file):
        print(f"\n❌ File not found: {audio_file}")
        return False
    
    print(f"\n🎵 Testing with: {os.path.basename(audio_file)}")
    print("=" * 70)
    
    try:
        # Run detection
        results = detect_gender_from_file(audio_file)
        
        print("\n✅ TEST PASSED! Gender detection is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    
    if len(sys.argv) < 2:
        print("\n📖 Usage: python quick_test.py <audio_file>")
        print("\nExample:")
        print("  python quick_test.py sample.wav")
        print("  python quick_test.py C:\\path\\to\\audio.mp3")
        
        # Ask for file path
        print("\n" + "=" * 70)
        audio_file = input("\nEnter audio file path (or press Enter to skip): ").strip()
        
        if not audio_file:
            print("\n⏭️  Skipping test. Provide an audio file to test.")
            sys.exit(0)
        
        # Remove quotes
        audio_file = audio_file.strip('"\'')
    else:
        audio_file = sys.argv[1]
    
    # Run test
    success = quick_test(audio_file)
    
    if success:
        print("\n" + "=" * 70)
        print(" 🎉 ALL SYSTEMS GO! Ready for gender detection!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Use test_gender.py for full testing")
        print("  - Import gender_detector in your code")
        print("  - See README.md for more examples")
    else:
        print("\n" + "=" * 70)
        print(" ⚠️  Test failed. Please check the error above.")
        print("=" * 70)
