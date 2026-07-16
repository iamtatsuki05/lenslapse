"""Sphinx configuration for the LensLapse documentation site.

Built by scripts/build_docs.sh (sphinx-apidoc regenerates docs/api/ from src/lenslapse first)
and deployed under https://iamtatsuki05.github.io/lenslapse/docs/ by deploy-pages.yml. The
existing Markdown guides in this directory are included as-is via myst-parser.
"""

project = "LensLapse"
author = "Tatsuki Okada"
copyright = "2026, Tatsuki Okada"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# autodoc imports lenslapse from the uv environment (editable install), so no sys.path hacks.
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Import server (and with it fastapi) before autodoc touches any module: documenting the
# members of a torch/pydantic-using module first leaves the interpreter in a state where a
# subsequent fresh `import fastapi` fails inside pydantic's schema generation
# (PydanticSchemaGenerationError on `__pydantic_extra__`, Sphinx 9.1). With the import done
# here, autodoc finds lenslapse.server in sys.modules and never re-imports fastapi.
import lenslapse.server  # noqa: E402,F401  (import-not-at-top / unused: needed for its side effect)

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

templates_path = []
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "LensLapse"
html_theme_options = {
    "source_repository": "https://github.com/iamtatsuki05/lenslapse",
    "source_branch": "main",
    "source_directory": "docs/",
}
