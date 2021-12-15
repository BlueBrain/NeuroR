# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------

project = 'NeuroR'
version = get_distribution('neuror').version
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_click.ext',
    'sphinx_autorun',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
suppress_warnings = ["ref.python"]
autosummary_generate = True
autosummary_imported_members = True
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx-bluebrain-theme'
html_title = 'NeuroR'
html_show_sourcelink = False
html_theme_options = {
    "repo_url": "https://github.com/BlueBrain/NeuroR/",
    "repo_name": "BlueBrain/NeuroR"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']



# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False
from sphinx.ext.autosummary import Autosummary

class AutosummaryOverride(Autosummary):
    """Extends Autosummary to ensure the nosignatures option is set."""

    def run(self):
        """Wrap the autodoc output in a div with autodoc class."""
        self.options["nosignatures"] = self.options.get("nosignatures", True)
        result = super(AutosummaryOverride, self).run()
        return result

def add_autosummary_override(app):
    """Override the autosummary definition to ensure no signatures."""
    if "sphinx.ext.autosummary" in app.extensions:
        app.add_directive("autosummary", AutosummaryOverride, override=True)

def allow_only_neuror(app, what, name, obj, skip, options):
    """Check that the member is part of neuror, exlude otherwise."""
    if what in {"module", "class", "exception", "function"} and "neuror" not in getattr(obj, "__module__", ""):
        return True
    return skip

def setup(app):
    app.connect('builder-inited', add_autosummary_override)
    app.connect('autodoc-skip-member', allow_only_neuror)
