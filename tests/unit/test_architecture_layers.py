import ast
import json
import logging
from pathlib import Path

from oviqs.adapters.datasets import JsonlDatasetAdapter
from oviqs.adapters.observability import JsonLogTraceSink
from oviqs.adapters.plugins import EntryPointPluginRegistry, InMemoryPluginRegistry
from oviqs.adapters.reporting import MarkdownReportRenderer
from oviqs.adapters.runners.factory import build_logits_runner
from oviqs.adapters.storage import LocalArtifactStorage
from oviqs.application.dto.reports import EvaluationReport, ReportRun
from oviqs.application.orchestration import EvaluationJob, InMemoryJobManager
from oviqs.application.services.evaluation import compute_likelihood_section
from oviqs.domain.samples import EvalSample
from oviqs.interfaces.grpc import evaluation_response_mapping, gpu_suite_request_from_mapping
from oviqs.interfaces.http import HealthResponse, LikelihoodRunRequest
from oviqs.platform.config import load_profile_settings
from oviqs.platform.security import PathPolicy
from oviqs.ports.runners import RunnerCapabilities


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_interface_application_and_port_layers_do_not_import_legacy_modules():
    root = Path(__file__).resolve().parents[2] / "src" / "oviqs"
    forbidden_roots = (
        "oviqs.adapters",
        "oviqs.aggregation",
        "oviqs.core",
        "oviqs.datasets",
        "oviqs.metrics",
        "oviqs.models",
        "oviqs.references",
        "oviqs.reporting",
        "oviqs.runners",
    )

    offenders = []
    checked_files = [root / "cli.py"]
    for layer in ("interfaces", "application", "ports"):
        checked_files.extend((root / layer).rglob("*.py"))

    for path in checked_files:
        for module in _imported_modules(path):
            if module.startswith(forbidden_roots):
                offenders.append(f"{path.relative_to(root)} imports {module}")

    assert offenders == []


def test_architecture_ports_accept_existing_dummy_runner():
    runner = build_logits_runner("dummy", "dummy", "CPU")
    capabilities = RunnerCapabilities(exposes_logits=True, supported_devices=("CPU",))

    sample = EvalSample(id="s1", task_type="likelihood", text="hello world")
    section = compute_likelihood_section(runner, [sample])

    assert capabilities.supported_devices == ("CPU",)
    assert section["status"] == "pass"
    assert section["num_tokens"] > 0


def test_domain_and_reporting_shims_preserve_report_contract():
    report = EvaluationReport(run=ReportRun(id="run1"), likelihood={"perplexity": 2.0})
    rendered = MarkdownReportRenderer().render(report.model_dump(mode="json"))

    assert "run1" in rendered
    assert "perplexity" in rendered


def test_local_storage_adapter(tmp_path: Path):
    storage = LocalArtifactStorage()
    target = tmp_path / "artifact.txt"

    storage.write_text(target, "content")

    assert storage.exists(target)
    assert storage.read_text(target) == "content"


def test_jsonl_dataset_adapter_roundtrip(tmp_path: Path):
    adapter = JsonlDatasetAdapter()
    target = tmp_path / "samples.jsonl"
    sample = EvalSample(id="s1", task_type="likelihood", text="hello")

    adapter.write_samples([sample], target)

    assert adapter.read_samples(target) == [sample]


def test_plugin_registries_expose_stable_lookup():
    registry = InMemoryPluginRegistry({"dummy": object})
    entrypoints = EntryPointPluginRegistry("oviqs.runners")

    assert "dummy" in list(registry.names())
    assert registry.get("dummy") is object
    assert "dummy" in list(entrypoints.names())


def test_http_and_grpc_contract_mappers(tmp_path: Path):
    request = LikelihoodRunRequest(
        model="dummy",
        dataset=tmp_path / "samples.jsonl",
        out=tmp_path / "report.json",
    )
    grpc_request = gpu_suite_request_from_mapping(
        {
            "model": "dummy",
            "dataset_uri": str(request.dataset),
            "report_uri": str(request.out),
        }
    )
    response = evaluation_response_mapping(
        {
            "run": {"id": "run1"},
            "summary": {"overall_status": "pass"},
            "likelihood": {"perplexity": 2.0, "status": "pass"},
        },
        report_uri=str(request.out),
    )

    assert HealthResponse().status == "ok"
    assert grpc_request.backend == "openvino-runtime"
    assert response["metrics"][0]["name"] == "likelihood.perplexity"


def test_config_loader_and_path_policy(tmp_path: Path):
    base = tmp_path / "base.yaml"
    profile = tmp_path / "gpu.yaml"
    base.write_text("profile: base\ndefaults:\n  device: CPU\n", encoding="utf-8")
    profile.write_text("profile: gpu\ndefaults:\n  backend: openvino-runtime\n", encoding="utf-8")

    settings = load_profile_settings(base, profile, {"quality_gates": {"required": True}})
    safe_path = PathPolicy(tmp_path).resolve_for_write(tmp_path / "reports" / "out.json")

    assert settings.profile == "gpu"
    assert settings.defaults == {"device": "CPU", "backend": "openvino-runtime"}
    assert settings.quality_gates["required"] is True
    assert safe_path.parent.exists()


def test_job_manager_and_json_log_sink(caplog):
    manager = InMemoryJobManager()
    logger = logging.getLogger("oviqs-test")
    sink = JsonLogTraceSink(logger)

    manager.create(EvaluationJob(id="job1", kind="likelihood"))
    manager.update_status("job1", "running")
    with caplog.at_level(logging.INFO, logger="oviqs-test"):
        sink.event("job_started", {"job_id": "job1"})

    payload = json.loads(caplog.records[0].message)
    assert manager.get("job1").status == "running"
    assert payload["event"] == "job_started"
