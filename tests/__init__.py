"""Test package. Point XDG_CONFIG_HOME at an empty folder so a real ~/.config/python_ollama/config.toml
on the developer's machine can never change a test result."""
import os
import tempfile

os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="ollama-test-config-")
