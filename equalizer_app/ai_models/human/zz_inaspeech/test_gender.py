from gender_detector import GenderDetector

if __name__ == "__main__":
    audio_file = r"C:\Users\amkma\Desktop\speeches\ai_separated\spanish_test\speaker_01_male_adult_vi__Vietnamese.wav"
    detector = GenderDetector()
    results = detector.detect(audio_file)
    gender = results['dominant_gender']
    print(gender)
