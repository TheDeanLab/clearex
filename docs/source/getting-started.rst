Getting Started
===============

Installation
------------

Install ClearEx in editable mode with documentation dependencies:

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install -e ".[docs]"

If you also need development and test tooling:

.. code-block:: bash

   uv pip install -e ".[dev,docs]"

Running ClearEx
---------------

Launch the GUI:

.. code-block:: bash

   uv run clearex --gui

Run in headless mode against an experiment file:

.. code-block:: bash

   uv run python -m clearex.main --headless --no-gui --file /path/to/experiment.yml --dask

Documentation Build
-------------------

Build HTML docs locally:

.. code-block:: bash

   uv run python -m sphinx -W --keep-going -b html docs/source docs/_build/html

Rendered output is written to ``docs/_build/html/index.html``.
