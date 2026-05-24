from risk.risk_applicant_profile import ApplicantProfile
from risk.risk_model_artifact import load_random_forest_artifact
from risk.risk_model_feature_builder import build_batch_model_features, build_model_features


def predict_model_default_probability(profile: ApplicantProfile) -> float | None:
    artifact = load_random_forest_artifact()
    if artifact is None:
        return None

    pipeline = artifact["pipeline"]
    features = build_model_features(profile)
    return float(pipeline.predict_proba(features)[:, 1][0])


def predict_model_default_probabilities(profiles: list[ApplicantProfile]) -> list[float] | None:
    artifact = load_random_forest_artifact()
    if artifact is None:
        return None
    if not profiles:
        return []

    pipeline = artifact["pipeline"]
    features = build_batch_model_features(profiles)
    return [float(value) for value in pipeline.predict_proba(features)[:, 1]]


def random_forest_artifact_available() -> bool:
    return load_random_forest_artifact() is not None
