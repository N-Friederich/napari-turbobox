Contributing
============

Bug reports and pull requests are welcome at
https://github.com/N-Friederich/napari-turbobox.

Development environment
-----------------------

Use Python 3.10 or newer and install napari with a Qt binding first, then the
package in editable mode:

.. code-block:: bash

   pip install "napari[all]>=0.6.6,<0.10"
   pip install -e ".[dev]"

The ``dev`` extra installs pytest, pytest-qt, hypothesis, pytest-cov, ruff,
npe2, build and twine (no Qt binding; ``napari[all]`` brings one). The
``testing`` extra (``pip install -e ".[testing]"``) installs only pytest,
pytest-qt and hypothesis and no Qt binding.

If more than one Qt binding is installed, qtpy (used by napari) and pytest-qt
may pick different ones. Select the binding explicitly: ``QT_API`` for qtpy and
``PYTEST_QT_API`` (or the ``qt_api`` option in ``pytest.ini``) for pytest-qt,
for example ``QT_API=pyqt6 PYTEST_QT_API=pyqt6 pytest``.

Running the tests
-----------------

.. code-block:: bash

   pytest

The tests create real napari viewers and therefore need a display. On a
headless Linux machine run them under a virtual X server (for example
``xvfb-run pytest``). The GitHub Actions workflow
(``.github/workflows/test.yml``) runs ``ruff check`` on ``src``, ``tests``,
``examples``, ``tutorials/tutorial_utils.py``, the demo recorder
``docs/make_demo.py`` and the benchmark core in
``paper_benchmarks/``, validates the
plugin manifest with ``npe2 validate``, and runs the tests on Linux (under
Xvfb), macOS and Windows with Python 3.10, 3.11 and 3.12, plus napari 0.6.6,
0.7.0 and 0.9.1 on Linux. For a version tag, ``.github/workflows/release.yml``
runs this workflow as well and publishes to PyPI only if all its jobs pass.

The test suite has been run locally with napari 0.6.6, 0.7.0 and 0.9.1.

Linting and pre-commit
----------------------

The ``pre-commit`` configuration runs the checks of the CI lint job (ruff on
the same files and with the same pinned version, and the plugin manifest check)
and a few file checks. It has no formatter hooks. Install and run it with:

.. code-block:: bash

   pip install pre-commit
   pre-commit install
   pre-commit run --all-files

README demo
-----------

The GIF at the top of the README (``docs/demo.gif``) and the two screenshots
further down (``docs/screenshot-panel.png``, ``docs/screenshot-views.png``) are
recorded from a real napari session by ``docs/make_demo.py``. The script
downloads the *C. elegans* volume of the first tutorial once (84 MB), moves one
box off its nucleus (see ``plan_scenes``), opens the four-viewer layout, drags
and resizes the box back onto its nucleus, undoes and redoes the fit with real
mouse and key events, and stops if any step does not produce the expected
result in the store and the views. It needs a display, ``ffmpeg`` and napari
with a Qt binding, and it is not part of the test suite. It uses some of
napari's private attributes and has been run with napari 0.7.0 and PyQt6 on
macOS. After a change to the plugin, record the files again and
commit them if they changed:

.. code-block:: bash

   python docs/make_demo.py
   # or through pre-commit (reports "files were modified" when they changed):
   pre-commit run demo-gif --hook-stage manual --all-files

Guidelines
----------

* Add a test for every behaviour change. Tests that need a viewer use napari's
  ``make_napari_viewer`` fixture.
* Changes to ``SpatialIndex`` must keep the property-based test
  (``tests/test_spatial_index_property.py``) passing.
* Keep documentation statements about behaviour and performance in line with
  the code and with measured results; do not document numbers that are not
  backed by a committed measurement.

Documentation
-------------

The documentation sources are in ``docs/`` (Sphinx with autodoc and napoleon).
There is no automated documentation build. To build it locally, in an
environment where napari and napari-TurboBox are installed (autodoc imports the
package):

.. code-block:: bash

   pip install sphinx
   sphinx-build -b html docs docs/_build/html
