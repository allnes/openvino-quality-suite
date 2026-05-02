import ast
import json
import logging
from importlib import import_module
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


def test_domain_interface_application_and_port_layers_do_not_import_legacy_modules():
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
        "oviqs.tokenization",
        "oviqs.utils",
    )

    offenders = []
    checked_files = [root / "cli.py"]
    for layer in ("domain", "interfaces", "application", "ports"):
        checked_files.extend((root / layer).rglob("*.py"))

    for path in checked_files:
        for module in _imported_modules(path):
            if module.startswith(forbidden_roots):
                offenders.append(f"{path.relative_to(root)} imports {module}")

    assert offenders == []


def test_legacy_package_roots_are_removed():
    root = Path(__file__).resolve().parents[2] / "src" / "oviqs"
    legacy_roots = (
        "aggregation",
        "core",
        "datasets",
        "integrations",
        "metrics",
        "models",
        "references",
        "reporting",
        "runners",
        "tokenization",
        "utils",
    )

    remaining = [name for name in legacy_roots if (root / name).exists()]

    assert remaining == []


def test_recommended_architecture_layout_files_exist():
    root = Path(__file__).resolve().parents[2] / "src" / "oviqs"
    required = [
        "interfaces/cli/app.py",
        "interfaces/cli/formatters.py",
        "interfaces/http/app.py",
        "interfaces/http/routes_runs.py",
        "interfaces/http/routes_models.py",
        "interfaces/http/routes_datasets.py",
        "interfaces/http/schemas.py",
        "interfaces/grpc/server.py",
        "interfaces/grpc/mappers.py",
        "application/dto/requests.py",
        "application/dto/responses.py",
        "application/services/evaluate_likelihood.py",
        "application/services/evaluate_drift.py",
        "application/services/evaluate_long_context.py",
        "application/services/evaluate_serving.py",
        "application/services/evaluate_rag.py",
        "application/services/evaluate_agent.py",
        "application/services/compare_reports.py",
        "application/services/render_report.py",
        "application/services/run_suite.py",
        "application/orchestration/job_manager.py",
        "application/orchestration/dependency_graph.py",
        "domain/reports/models.py",
        "domain/reports/status.py",
        "domain/metrics/likelihood.py",
        "domain/metrics/distribution_drift.py",
        "domain/metrics/long_context.py",
        "domain/metrics/generation.py",
        "domain/metrics/serving.py",
        "domain/metrics/rag.py",
        "domain/metrics/agent.py",
        "domain/metrics/performance.py",
        "domain/gates/evaluator.py",
        "domain/gates/thresholds.py",
        "domain/references/catalog.py",
        "domain/references/oracles.py",
        "domain/traces/models.py",
        "domain/samples/models.py",
        "domain/samples/validation.py",
        "ports/runners.py",
        "ports/tokenizers.py",
        "ports/datasets.py",
        "ports/storage.py",
        "ports/reporting.py",
        "ports/model_registry.py",
        "ports/observability.py",
        "ports/plugins.py",
        "adapters/runners/dummy.py",
        "adapters/runners/hf.py",
        "adapters/runners/optimum_openvino.py",
        "adapters/runners/ov_runtime.py",
        "adapters/runners/ov_genai.py",
        "adapters/runners/ovms.py",
        "adapters/tokenizers/hf_tokenizer.py",
        "adapters/tokenizers/ov_tokenizer.py",
        "adapters/datasets/jsonl.py",
        "adapters/datasets/longbench.py",
        "adapters/datasets/helmet.py",
        "adapters/datasets/ruler.py",
        "adapters/storage/filesystem.py",
        "adapters/storage/sqlite_index.py",
        "adapters/storage/postgres_index.py",
        "adapters/storage/s3_artifacts.py",
        "adapters/reporting/json_report.py",
        "adapters/reporting/markdown_report.py",
        "adapters/reporting/html_report.py",
        "adapters/reporting/plot_report.py",
        "adapters/observability/logging_json.py",
        "adapters/observability/otel.py",
        "adapters/plugins/entrypoints.py",
        "contracts/jsonschema/evaluation_report.schema.json",
        "contracts/jsonschema/eval_request.schema.json",
        "contracts/proto/evaluation.proto",
        "platform/bootstrap/container.py",
        "platform/bootstrap/factories.py",
        "platform/security/path_policy.py",
        "platform/security/secrets.py",
    ]

    missing = [path for path in required if not (root / path).exists()]

    assert missing == []


def test_removed_internal_adapter_module_names_do_not_reappear():
    root = Path(__file__).resolve().parents[2] / "src" / "oviqs"
    removed = [
        "adapters/reporting/json.py",
        "adapters/reporting/markdown.py",
        "adapters/reporting/html.py",
        "adapters/reporting/plots.py",
        "adapters/storage/local.py",
        "adapters/tokenizers/tokenizer.py",
        "adapters/tokenizers/chat_template.py",
        "adapters/tokenizers/hashing.py",
        "application/services/reports.py",
    ]

    remaining = [path for path in removed if (root / path).exists()]

    assert remaining == []


def test_domain_does_not_parse_yaml_files():
    root = Path(__file__).resolve().parents[2] / "src" / "oviqs" / "domain"
    offenders = []
    for path in root.rglob("*.py"):
        for module in _imported_modules(path):
            if module == "yaml":
                offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_builtin_runner_classes_publish_capabilities():
    classes = [
        ("oviqs.adapters.runners.dummy", "DummyLogitsRunner", "exposes_logits"),
        ("oviqs.adapters.runners.dummy", "DummyGenerationRunner", "supports_generation"),
        ("oviqs.adapters.runners.hf", "HFReferenceRunner", "exposes_logits"),
        ("oviqs.adapters.runners.ov_runtime", "OVRuntimeLogitsRunner", "exposes_logits"),
        ("oviqs.adapters.runners.optimum_openvino", "OptimumOVLogitsRunner", "exposes_logits"),
        ("oviqs.adapters.runners.ov_genai", "OVGenAIRunner", "supports_generation"),
        ("oviqs.adapters.runners.ovms", "OVMSOpenAIRunner", "supports_generation"),
    ]

    for module_name, class_name, expected_flag in classes:
        cls = getattr(import_module(module_name), class_name)
        capabilities = cls.capabilities
        assert getattr(capabilities, expected_flag) is True
        assert capabilities.supported_devices


def test_architecture_ports_accept_existing_dummy_runner():
    runner = build_logits_runner("dummy", "dummy", "CPU")
    capabilities = RunnerCapabilities(exposes_logits=True, supported_devices=("CPU",))

    sample = EvalSample(id="s1", task_type="likelihood", text="hello world")
    section = compute_likelihood_section(runner, [sample])

    assert capabilities.supported_devices == ("CPU",)
    assert section["status"] == "pass"
    assert section["num_tokens"] > 0


def test_domain_report_and_reporting_adapter_preserve_report_contract():
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
