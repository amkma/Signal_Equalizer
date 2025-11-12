from gender_detector import GenderDetector

if __name__ == "__main__":
    audio_file = r"C:\Users\amkma\Desktop\speeches\ai_separated\spanish_test\speaker_02_female_adult_es__Spanish.wav"
    detector = GenderDetector()
    results = detector.detect(audio_file)
    gender = results['dominant_gender']
    print(gender)
