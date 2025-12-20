"""Architecture boundary tests.

Enforce module dependency invariants to prevent architectural drift.
"""

import ast
from pathlib import Path


def _imports_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    return imports


def _assert_no_imports(paths: list[Path], *, banned_prefixes: tuple[str, ...]) -> None:
    violations = [
        f"{path}: imports {module}"
        for path in paths
        for module in sorted(_imports_in_file(path))
        if module.startswith(banned_prefixes)
    ]
    assert not violations, "Boundary violations:\n" + "\n".join(violations)


def test_modules_are_pure() -> None:
    """foundry/modules/ should not import from cli, train, or config."""
    modules = sorted(Path("foundry/modules").glob("*.py"))
    _assert_no_imports(modules, banned_prefixes=("foundry.cli", "foundry.train", "foundry.config"))


def test_data_does_not_import_train() -> None:
    """foundry/data/ should not import training loop."""
    data = sorted(Path("foundry/data").glob("*.py"))
    _assert_no_imports(data, banned_prefixes=("foundry.train",))


def test_model_is_standalone() -> None:
    """foundry/model.py should not import from cli, train, or data."""
    model = [Path("foundry/model.py")]
    _assert_no_imports(model, banned_prefixes=("foundry.cli", "foundry.train", "foundry.data"))


def test_cli_does_not_import_modules_internals() -> None:
    """foundry/cli/ should not reach into modules/ internals (use model.py)."""
    cli = sorted(Path("foundry/cli").glob("*.py"))
    _assert_no_imports(cli, banned_prefixes=("foundry.modules.",))
