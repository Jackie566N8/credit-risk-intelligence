from functools import lru_cache

import joblib

from config.app_paths import MODEL_ARTIFACT_FILES


@lru_cache(maxsize=1)
def load_random_forest_artifact() -> dict | None:
    artifact_path = MODEL_ARTIFACT_FILES["random_forest"]
    if not artifact_path.exists():
        return None
    return joblib.load(artifact_path)
