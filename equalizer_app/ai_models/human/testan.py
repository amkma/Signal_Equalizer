from ai_voice_separator import AIVoiceSeparator

separator = AIVoiceSeparator()

input_file = r"C:\Users\user\Desktop\test_01.wav"
output_dir = r"C:\Users\user\Desktop\test_human_output"

separator.run_pipeline(input_file, output_dir)
