"""Tests use the archived Línea Directa KB (with its Block H questions) as a fixture, not the live KB."""
import os

os.environ.setdefault("KB_CONFIG", "kb/archive/linea_directa.yaml")
