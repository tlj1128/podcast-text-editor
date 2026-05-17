import json
from pathlib import Path


class PodcastFileManager:
    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        self.config_path = Path("config.json")          # 專案根目錄
        self.session_path = self.base_dir / "session.json"  # 自動存檔

    def get_project_dir(self, audio_path: str) -> Path:
        audio_name = Path(audio_path).stem
        project_dir = self.base_dir / audio_name
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def get_episode_session_path(self, audio_path: str) -> Path:
        return self.get_project_dir(audio_path) / "session.json"

    def save_json(self, file_path: Path, data: dict):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, file_path: Path) -> dict:
        if not file_path.exists():
            return {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def load_config(self) -> dict:
        return self.load_json(self.config_path)

    def save_config(self, data: dict):
        self.save_json(self.config_path, data)
