Getting Started
===============

Installation
------------

ClearEx currently targets Python ``3.12`` (project constraint:
``>=3.12,<3.13``). Using ``uv`` with an explicit Python version avoids
accidental unsupported environments.

macOS
^^^^^

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.12
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e ".[docs]"

Linux
^^^^^

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.12
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -e ".[docs]"

Windows (PowerShell)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: powershell

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   uv python install 3.12
   uv venv --python 3.12
   .venv\Scripts\Activate.ps1
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

Run in headless mode with an explicit Navigate stage-to-world mapping:

.. code-block:: bash

   uv run clearex --headless \
     --file /path/to/experiment.yml \
     --visualization \
     --stage-axis-map z=+x,y=none,x=+y

In the GUI setup flow, the same mapping can be authored through the
``Spatial Calibration`` panel before entering analysis selection.

Documentation Build
-------------------

Build HTML docs locally:

.. code-block:: bash

   uv run python -m sphinx -W --keep-going -b html docs/source docs/_build/html

Rendered output is written to ``docs/_build/html/index.html``.
