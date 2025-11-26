from ai_voice_separator import AIVoiceSeparator

separator = AIVoiceSeparator()

input_file = r"c:\Users\amkma\Desktop\naknan\my_4_speaker_mix_48k.wav"
output_dir = r"C:\Users\amkma\Desktop\speeches\ai_separated\spanish_test"

separator.run_pipeline(input_file, output_dir)
