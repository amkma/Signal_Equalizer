import subprocess
import json
from pathlib import Path
from typing import Dict


class AIOrchestrator:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.venv_base = self.base_dir.parent / ".venvan"
        self.human_venv = self.venv_base / ".human_sep_venv"
        self.music_venv = self.venv_base / ".music_venv"
        self.human_script = self.base_dir / "human" / "ai_voice_separator.py"
        self.music_script = self.base_dir / "music" / "music_separator.py"
        self._validate_setup()

    def _validate_setup(self):
        if not self.human_venv.exists():
            raise RuntimeError(f"Human venv not found: {self.human_venv}")
        if not self.music_venv.exists():
            raise RuntimeError(f"Music venv not found: {self.music_venv}")
        if not self.human_script.exists():
            raise RuntimeError(f"Human script not found: {self.human_script}")
        if not self.music_script.exists():
            raise RuntimeError(f"Music script not found: {self.music_script}")

    def separate_human_voices(self, audio_path: str, output_dir: str) -> Dict:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        python_exe = self.human_venv / "Scripts" / "python.exe"
        
        script_code = f"""
import sys
sys.path.insert(0, r"{self.human_script.parent}")
from ai_voice_separator import AIVoiceSeparator
separator = AIVoiceSeparator()
separator.run_pipeline(r"{audio_path}", r"{output_dir}")
"""
        
        result = subprocess.run(
            [str(python_exe), "-c", script_code],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Separation failed: {result.stderr}")
        
        metadata_path = Path(output_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        return {"status": "completed", "output_dir": output_dir}

    def separate_music(self, audio_path: str, output_dir: str, model: str = "spleeter:4stems") -> Dict:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        python_exe = self.music_venv / "Scripts" / "python.exe"
        
        script_code = f"""
import sys
import json
sys.path.insert(0, r"{self.music_script.parent}")
from music_separator import MusicSeparator
separator = MusicSeparator(model="{model}")
result = separator.separate(r"{audio_path}", r"{output_dir}")
print("__RESULT__")
print(json.dumps(result, indent=2))
"""
        
        result = subprocess.run(
            [str(python_exe), "-c", script_code],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Separation failed: {result.stderr}")
        
        output_lines = result.stdout.split("\n")
        try:
            result_idx = output_lines.index("__RESULT__")
            result_json = "\n".join(output_lines[result_idx + 1:])
            return json.loads(result_json)
        except (ValueError, json.JSONDecodeError):
            return {"status": "completed", "output_dir": output_dir}

def separate_human_voices(audio_path: str, output_dir: str) -> Dict:
    return AIOrchestrator().separate_human_voices(audio_path, output_dir)


def separate_music(audio_path: str, output_dir: str, model: str = "spleeter:4stems") -> Dict:
    return AIOrchestrator().separate_music(audio_path, output_dir, model)

if __name__ == "__main__":
    orchy = AIOrchestrator()
    orchy.separate_human_voices(r"C:\Users\amkma\Desktop\naknan\test_01.wav",
                                r"C:\Users\amkma\Desktop\separated_sources")
    # orchy.separate_music(r"C:\Users\amkma\Desktop\505.wav",
    #                      r"C:\Users\amkma\Desktop\testy")