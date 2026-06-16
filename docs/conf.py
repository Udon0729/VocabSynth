"""Sphinx 設定。"""

project = "VocabSynth"
copyright = "2026"
author = "kmunaoka"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "furo"

exclude_patterns = ["_build"]
