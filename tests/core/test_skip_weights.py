"""Tests for the skip_weights flag when debug_guidance is enabled.

Verifies that:
1. start_policy_server builds the correct CLI command with --skip-weights
2. load_pytorch implementations guard weight loading behind skip_weights
3. create_trained_policy accepts and threads skip_weights
4. serve_policy.py Args dataclass has skip_weights field

Tests 2-4 use AST/source inspection since the openpi modules have heavy
dependencies not available in the main project venv.
"""

import ast
import inspect
import pathlib
import unittest
from unittest.mock import MagicMock, patch

# Resolve project root (tests/core/test_skip_weights.py -> project root)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_SRC = PROJECT_ROOT / "third_party" / "openpi" / "src"
OPENPI_SCRIPTS = PROJECT_ROOT / "third_party" / "openpi" / "scripts"


def _read_source(path: pathlib.Path | str) -> str:
    with open(path) as f:
        return f.read()


def _parse_ast(path: pathlib.Path | str) -> ast.Module:
    return ast.parse(_read_source(path))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _get_param_names(func_node: ast.FunctionDef) -> list[str]:
    """Return all parameter names from a function def, including kwonly."""
    names = [a.arg for a in func_node.args.args]
    names += [a.arg for a in func_node.args.kwonlyargs]
    return names


# =========================================================================
# 1. Test: start_policy_server CLI construction (live import)
# =========================================================================


class TestStartPolicyServerCommand(unittest.TestCase):
    """Verify start_policy_server appends --skip-weights when requested."""

    @patch("subprocess.Popen")
    def test_skip_weights_true_appends_flag(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=12345)
        from policy_abstraction.policies.openpi import start_policy_server

        start_policy_server(port=8000, skip_weights=True)
        cmd = mock_popen.call_args[0][0]
        self.assertIn("--skip-weights", cmd)

    @patch("subprocess.Popen")
    def test_skip_weights_before_subcommand(self, mock_popen):
        """--skip-weights must appear before the policy:checkpoint subcommand (tyro ordering)."""
        mock_popen.return_value = MagicMock(pid=12345)
        from policy_abstraction.policies.openpi import start_policy_server

        start_policy_server(port=8000, skip_weights=True)
        cmd = mock_popen.call_args[0][0]
        skip_idx = cmd.index("--skip-weights")
        subcmd_idx = cmd.index("policy:checkpoint")
        self.assertLess(
            skip_idx,
            subcmd_idx,
            "--skip-weights must come before policy:checkpoint subcommand",
        )

    @patch("subprocess.Popen")
    def test_skip_weights_false_omits_flag(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=12345)
        from policy_abstraction.policies.openpi import start_policy_server

        start_policy_server(port=8000, skip_weights=False)
        cmd = mock_popen.call_args[0][0]
        self.assertNotIn("--skip-weights", cmd)

    @patch("subprocess.Popen")
    def test_skip_weights_default_omits_flag(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=12345)
        from policy_abstraction.policies.openpi import start_policy_server

        start_policy_server(port=8000)
        cmd = mock_popen.call_args[0][0]
        self.assertNotIn("--skip-weights", cmd)

    @patch("subprocess.Popen")
    def test_signature_has_skip_weights_defaulting_false(self, mock_popen):
        from policy_abstraction.policies.openpi import start_policy_server

        sig = inspect.signature(start_policy_server)
        param = sig.parameters.get("skip_weights")
        self.assertIsNotNone(param, "skip_weights parameter missing from signature")
        assert param is not None
        self.assertEqual(param.default, False)


# =========================================================================
# 2. Test: model.py base load_pytorch has skip_weights guard
# =========================================================================


class TestModelLoadPytorchSource(unittest.TestCase):
    """Verify base BaseModelConfig.load_pytorch guards safetensors via skip_weights."""

    MODEL_PATH = OPENPI_SRC / "openpi" / "models" / "model.py"

    def setUp(self):
        self.tree = _parse_ast(self.MODEL_PATH)
        self.source = _read_source(self.MODEL_PATH)

    def test_load_pytorch_has_skip_weights_param(self):
        cls = _find_class(self.tree, "BaseModelConfig")
        self.assertIsNotNone(cls, "BaseModelConfig class not found")
        assert cls is not None
        func = None
        for node in ast.walk(cls):
            if isinstance(node, ast.FunctionDef) and node.name == "load_pytorch":
                func = node
                break
        self.assertIsNotNone(func, "load_pytorch method not found in BaseModelConfig")
        assert func is not None
        params = _get_param_names(func)
        self.assertIn("skip_weights", params)

    def test_load_pytorch_guards_load_model(self):
        lines = self.source.splitlines()
        load_model_lines = [
            i
            for i, line in enumerate(lines)
            if "load_model" in line and not line.lstrip().startswith("#")
        ]
        guard_found = False
        for idx in load_model_lines:
            context = "\n".join(lines[max(0, idx - 3) : idx + 1])
            if "not skip_weights" in context:
                guard_found = True
                break
        self.assertTrue(
            guard_found,
            "Expected 'if not skip_weights' guard before safetensors.torch.load_model",
        )


# =========================================================================
# 3. Test: pi0grounded_config.py load_pytorch has skip_weights guard
# =========================================================================


class TestGroundedConfigLoadPytorchSource(unittest.TestCase):
    """Verify Pi0GroundedConfig.load_pytorch guards safetensors via skip_weights."""

    CONFIG_PATH = OPENPI_SRC / "openpi" / "models" / "pi0grounded_config.py"

    def setUp(self):
        self.tree = _parse_ast(self.CONFIG_PATH)
        self.source = _read_source(self.CONFIG_PATH)

    def test_load_pytorch_has_skip_weights_param(self):
        cls = _find_class(self.tree, "Pi0GroundedConfig")
        self.assertIsNotNone(cls)
        assert cls is not None
        func = None
        for node in ast.walk(cls):
            if isinstance(node, ast.FunctionDef) and node.name == "load_pytorch":
                func = node
                break
        self.assertIsNotNone(func, "load_pytorch not found in Pi0GroundedConfig")
        assert func is not None
        params = _get_param_names(func)
        self.assertIn("skip_weights", params)

    def test_load_pytorch_guards_load_model(self):
        lines = self.source.splitlines()
        load_model_lines = [
            i
            for i, line in enumerate(lines)
            if "load_model" in line and not line.lstrip().startswith("#")
        ]
        guard_found = False
        for idx in load_model_lines:
            context = "\n".join(lines[max(0, idx - 3) : idx + 1])
            if "not skip_weights" in context:
                guard_found = True
                break
        self.assertTrue(
            guard_found,
            "Expected 'if not skip_weights' guard before safetensors.torch.load_model",
        )


# =========================================================================
# 4. Test: create_trained_policy has skip_weights param
# =========================================================================


class TestCreateTrainedPolicySource(unittest.TestCase):
    """Verify create_trained_policy accepts and forwards skip_weights."""

    POLICY_CONFIG_PATH = OPENPI_SRC / "openpi" / "policies" / "policy_config.py"

    def setUp(self):
        self.tree = _parse_ast(self.POLICY_CONFIG_PATH)
        self.source = _read_source(self.POLICY_CONFIG_PATH)

    def test_has_skip_weights_param(self):
        func = _find_function(self.tree, "create_trained_policy")
        self.assertIsNotNone(func)
        assert func is not None
        params = _get_param_names(func)
        self.assertIn("skip_weights", params)

    def test_passes_skip_weights_to_load_pytorch(self):
        self.assertIn(
            "skip_weights=skip_weights",
            self.source,
            "create_trained_policy should pass skip_weights=skip_weights to load_pytorch",
        )

    def test_jax_path_uses_create_when_skipping(self):
        self.assertIn(
            "train_config.model.create(",
            self.source,
            "JAX skip_weights path should call config.create()",
        )


# =========================================================================
# 5. Test: serve_policy.py Args has skip_weights field
# =========================================================================


class TestServePolicyArgsSource(unittest.TestCase):
    """Verify serve_policy.py Args dataclass includes skip_weights."""

    SERVE_PATH = OPENPI_SCRIPTS / "serve_policy.py"

    def setUp(self):
        self.tree = _parse_ast(self.SERVE_PATH)
        self.source = _read_source(self.SERVE_PATH)

    def test_args_has_skip_weights_field(self):
        cls = _find_class(self.tree, "Args")
        self.assertIsNotNone(cls, "Args class not found in serve_policy.py")
        assert cls is not None
        field_names = [
            node.target.id
            for node in ast.walk(cls)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        ]
        self.assertIn("skip_weights", field_names)

    def test_create_policy_forwards_skip_weights(self):
        self.assertIn(
            "skip_weights=args.skip_weights",
            self.source,
            "create_policy should forward args.skip_weights",
        )


if __name__ == "__main__":
    unittest.main()
