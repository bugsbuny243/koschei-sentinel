from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from koschei_sentinel.dataset import DatasetExample
from koschei_sentinel.models import StrictModel
from koschei_sentinel.policy import baseline_opinion, validate_opinion
from koschei_sentinel.readiness import DatasetReadinessReport

_SPLIT_NAMES = ("train", "validation", "test")
SYSTEM_PROMPT = (
    "You are Koschei Sentinel. Return exactly one JSON object matching "
    "sentinel.opinion.v1. The signed deterministic verdict is final. "
    "Every factual claim must cite supplied evidence IDs. Never invent evidence, "
    "raise confidence, expose personal data, or replace the deterministic assessment."
)


class QuantizationConfig(StrictModel):
    bits: Literal[4, 8] = 4
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"


class LoraConfig(StrictModel):
    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1, le=1024)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        min_length=1,
        max_length=32,
    )

    @model_validator(mode="after")
    def target_modules_are_safe(self) -> LoraConfig:
        if len(self.target_modules) != len(set(self.target_modules)):
            raise ValueError("target_modules must be unique")
        if any(not item or "/" in item or "\\" in item for item in self.target_modules):
            raise ValueError("target_modules must contain plain module names")
        return self


class TrainingConfig(StrictModel):
    schema_version: Literal["sentinel.training-config.v1"] = "sentinel.training-config.v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    base_model: str = Field(min_length=3, max_length=256)
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    dataset_release: str = Field(min_length=1, max_length=1024)
    readiness_report: str | None = Field(default=None, min_length=1, max_length=1024)
    require_readiness: bool = False
    output_dir: str = Field(min_length=1, max_length=1024)
    max_sequence_length: int = Field(default=2048, ge=256, le=32768)
    epochs: float = Field(default=1.0, gt=0.0, le=20.0)
    learning_rate: float = Field(default=0.0002, gt=0.0, le=0.1)
    per_device_batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=16, ge=1, le=1024)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    logging_steps: int = Field(default=5, ge=1, le=10000)
    seed: int = Field(default=1701, ge=0, le=2**31 - 1)
    trust_remote_code: Literal[False] = False
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)

    @model_validator(mode="after")
    def inputs_are_safe(self) -> TrainingConfig:
        _validate_relative_path(self.dataset_release, "dataset_release")
        _validate_relative_path(self.output_dir, "output_dir")
        if self.readiness_report is not None:
            _validate_relative_path(self.readiness_report, "readiness_report")
        if self.require_readiness and self.readiness_report is None:
            raise ValueError("require_readiness requires readiness_report")
        if self.base_model.startswith(("http://", "https://")):
            raise ValueError("base_model must be a registry identifier, not a URL")
        if self.base_model.count("/") != 1:
            raise ValueError("base_model must use owner/model format")
        return self

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


class TrainingSplit(StrictModel):
    path: str
    examples: int
    groups: int
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class TrainingPlan(StrictModel):
    schema_version: Literal["sentinel.training-plan.v1"] = "sentinel.training-plan.v1"
    dry_run: bool = True
    run_id: str
    base_model: str
    base_revision: str
    dataset_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    readiness_report_digest: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    training_config_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    splits: dict[str, TrainingSplit]
    effective_batch_size: int
    estimated_optimizer_steps: int
    output_dir: str
    warnings: list[str] = Field(default_factory=list)


class AdapterManifest(StrictModel):
    schema_version: Literal["sentinel.adapter-manifest.v1"] = "sentinel.adapter-manifest.v1"
    run_id: str
    base_model: str
    base_revision: str
    dataset_manifest_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_config_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_files: list[str] = Field(min_length=1, max_length=256)
    output_dir: str


def load_training_config(path: str | Path) -> TrainingConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("training config is not valid JSON") from exc
    return TrainingConfig.model_validate(payload)


def plan_training(config: TrainingConfig, *, root: str | Path = ".") -> TrainingPlan:
    root_path = Path(root).resolve()
    release_path = resolve_under_root(root_path, config.dataset_release)
    manifest_path = release_path / "quality-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("dataset release is missing quality-manifest.json")

    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("dataset quality manifest is not valid JSON") from exc
    if manifest.get("schema_version") != "sentinel.quality-manifest.v1":
        raise ValueError("unsupported dataset quality manifest schema")
    if manifest.get("dry_run") is not False:
        raise ValueError("training requires a materialized non-dry-run dataset release")
    reports = manifest.get("splits")
    if not isinstance(reports, dict):
        raise ValueError("dataset quality manifest is missing split reports")

    splits: dict[str, TrainingSplit] = {}
    warnings: list[str] = []
    for split_name in _SPLIT_NAMES:
        report = reports.get(split_name)
        if not isinstance(report, dict):
            raise ValueError(f"dataset manifest is missing {split_name} split")
        relative_path = str(
            PurePosixPath(config.dataset_release) / f"{split_name}.jsonl"
        )
        rows, digest = load_release_examples(root_path, relative_path)
        if digest != report.get("digest"):
            raise ValueError(f"{split_name} split digest does not match quality manifest")
        if len(rows) != report.get("examples"):
            raise ValueError(f"{split_name} split count does not match quality manifest")
        groups = len({item.group_ref for item in rows})
        if groups != report.get("groups"):
            raise ValueError(f"{split_name} group count does not match quality manifest")
        if not rows:
            warnings.append(f"{split_name} split is empty")
        splits[split_name] = TrainingSplit(
            path=relative_path,
            examples=len(rows),
            groups=groups,
            digest=digest,
        )

    readiness_digest = _validate_readiness_report(
        config,
        root_path=root_path,
        release_path=release_path,
        manifest_digest=manifest_digest,
        warnings=warnings,
    )
    train_examples = splits["train"].examples
    if train_examples == 0:
        raise ValueError("train split must contain at least one example")
    steps_per_epoch = max(
        1,
        (train_examples + config.effective_batch_size - 1) // config.effective_batch_size,
    )
    estimated_steps = max(1, int(steps_per_epoch * config.epochs + 0.999999))
    if resolve_under_root(root_path, config.output_dir).exists():
        raise FileExistsError(f"training output already exists: {config.output_dir}")

    return TrainingPlan(
        run_id=config.run_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        dataset_manifest_digest=manifest_digest,
        readiness_report_digest=readiness_digest,
        training_config_digest=model_digest(config),
        splits=splits,
        effective_batch_size=config.effective_batch_size,
        estimated_optimizer_steps=estimated_steps,
        output_dir=config.output_dir,
        warnings=warnings,
    )


def supervised_messages(example: DatasetExample) -> list[dict[str, str]]:
    opinion = baseline_opinion(example.case)
    if validate_opinion(example.case, opinion):
        raise ValueError("generated supervision violates the Sentinel evidence policy")
    opinion_payload = opinion.model_dump(mode="json")
    opinion_payload.pop("generated_at", None)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(example.case.model_dump(mode="json"))},
        {"role": "assistant", "content": canonical_json(opinion_payload)},
    ]


def load_release_examples(
    root: Path, relative_path: str
) -> tuple[list[DatasetExample], str]:
    path = resolve_under_root(root, relative_path)
    if not path.is_file():
        raise ValueError(f"dataset release is missing {Path(relative_path).name}")
    raw = path.read_bytes()
    rows: list[DatasetExample] = []
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} is not valid UTF-8") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(DatasetExample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"invalid {path.stem} dataset row at line {line_number}"
            ) from exc
    return rows, hashlib.sha256(raw).hexdigest()


def write_training_plan(plan: TrainingPlan, path: str | Path) -> None:
    payload = json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    atomic_write(Path(path), payload)


def model_digest(model: StrictModel) -> str:
    return hashlib.sha256(canonical_json(model.model_dump(mode="json")).encode()).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resolve_under_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes the repository root")
    return candidate


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_readiness_report(
    config: TrainingConfig,
    *,
    root_path: Path,
    release_path: Path,
    manifest_digest: str,
    warnings: list[str],
) -> str | None:
    if config.readiness_report is None:
        return None
    report_path = resolve_under_root(root_path, config.readiness_report)
    if report_path.parent != release_path:
        raise ValueError("readiness_report must be stored inside dataset_release")
    if not report_path.is_file():
        raise ValueError("dataset release is missing readiness report")
    report_bytes = report_path.read_bytes()
    try:
        report = DatasetReadinessReport.model_validate_json(report_bytes)
    except ValueError as exc:
        raise ValueError("dataset readiness report is invalid") from exc
    if report.release_manifest_digest != manifest_digest:
        raise ValueError("readiness report does not match dataset quality manifest")
    if config.require_readiness and not report.ready:
        raise ValueError("dataset readiness report did not pass")
    if not report.ready:
        warnings.append("dataset readiness report did not pass")
    return hashlib.sha256(report_bytes).hexdigest()


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ValueError(f"{field} must stay within the repository root")
