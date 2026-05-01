from __future__ import annotations

from oviqs.adapters.runners.dummy import DummyLogitsRunner
from oviqs.ports.runners import GenerationRunnerPort, LogitsRunnerPort


def build_logits_runner(backend: str, model: str, device: str) -> LogitsRunnerPort:
    if backend == "dummy":
        return DummyLogitsRunner()
    if backend == "hf":
        from oviqs.adapters.runners.hf import HFReferenceRunner

        return HFReferenceRunner(model, device=device.lower())
    if backend == "optimum-openvino":
        from oviqs.adapters.runners.optimum_openvino import OptimumOVLogitsRunner

        return OptimumOVLogitsRunner(model, device=device)
    if backend == "openvino-runtime":
        from oviqs.adapters.runners.ov_runtime import OVRuntimeLogitsRunner

        return OVRuntimeLogitsRunner(model, device=device)
    raise ValueError(f"Unsupported backend: {backend}")


def build_generation_runner(backend: str, model: str, device: str) -> GenerationRunnerPort:
    if backend == "dummy":
        from oviqs.adapters.runners.dummy import DummyGenerationRunner

        return DummyGenerationRunner()
    if backend == "hf":
        from oviqs.adapters.runners.hf import HFReferenceRunner

        return HFReferenceRunner(model, device=device.lower())
    if backend == "optimum-openvino":
        from oviqs.adapters.runners.optimum_openvino import OptimumOVLogitsRunner

        return OptimumOVLogitsRunner(model, device=device)
    if backend == "openvino-genai":
        from oviqs.adapters.runners.ov_genai import OVGenAIRunner

        return OVGenAIRunner(model, device=device)
    if backend == "ovms-openai":
        from oviqs.adapters.runners.ovms import OVMSOpenAIRunner

        return OVMSOpenAIRunner(base_url=model, model=model)
    raise ValueError(f"Unsupported generation backend: {backend}")


__all__ = ["build_generation_runner", "build_logits_runner"]
