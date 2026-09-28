"""The runtime package must remain independent of research orchestration."""
import ast
from pathlib import Path


def test_core_does_not_import_experiment_or_observability_packages():
    core = Path(__file__).resolve().parents[1] / "src" / "atfm"
    violations = []
    for path in core.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.module == "atfm":
                    modules.extend(f"atfm.{alias.name}" for alias in node.names)
            else:
                continue
            for module in modules:
                if any(module == prefix or module.startswith(prefix + ".")
                       for prefix in ("atfm_experiments", "atfm.experiments", "phoenix", "openinference", "opentelemetry")):
                    violations.append(f"{path.relative_to(core)}:{node.lineno}: {module}")
    assert not violations, "Core imports experiment/observability packages:\n" + "\n".join(violations)
