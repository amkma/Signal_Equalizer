from ai_voice_separator import AIVoiceSeparator

separator = AIVoiceSeparator()

input_file = r"C:\Users\amkma\Desktop\naknan\test_01.wav"
output_dir = r"C:\Users\amkma\Desktop\speeches\ai_separated\spanish_test"

separator.run_pipeline(input_file, output_dir)
