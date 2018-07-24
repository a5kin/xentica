# Changelog
All notable changes to Xentica project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2018-07-24
### Changed

- Optimization.
  - Optimize ``BigBang``/``PrimordialSoup`` seed generation.
- Examples.
  - Add Shifting Sands example.
- Refactoring.
  - Make some private properties public.
  - Rename ``LocalRandom`` streams.
  - Add properties getter to ``ContainerProperty``.
  - Move GPU stuff to separate classes.
  - Decompose ``BSCA`` class creation into smaller methods.
  - Move ``core.topology.mixins`` to ``core.mixins``.

### Added

- QA.
  - Add pylint config.
  - Add pylint run script.
- Tests.
  - Add test for non-uniform buffer interactions.
  - Add test for unset viewport.

### Fixed

- Major fixes.
  - Fix direct buffer assignments.
  - Fix property assignments.
  - Fix direct Variable assignments.
- Minor fixes.
  - Fix ``Lattice.index_to_coord()``
  - Fix ``Experiment.seed.generate()`` arguments.
  - Fix ``RandInt`` to generate NumPy arrays.
- Refactoring.
  - Fix most of ``pylint`` issues through the code.
- Tests.
  - Fix ``BigBang``/``PrimordialSoup`` tests.
  - Fix checksums in tests.
- Miscellaneous.
  - Fix typos in docstrings.

## 0.1.0 - 2018-05-05
### Added

- Base functionality.
  - Translate Python functions describing CA logic into CUDA C.
  - Evolve CA step by step, using resulting GPU kernels.
  - Render the current CA state into 2D NumPy array for subsequent
    visualization.
  - Save and load current CA state.
- Multi-dimensional CA topologies.
  - N-dimensional orthogonal lattice.
  - N-dimensional cases of Moore and Von Neumann neighborhoods.
  - Wrap the board into N-torus.
  - Surround the board by static cells.
- CA inner logic.
  - Multiple integer 'properties' per each cell state.
  - Buffered interaction between neighbors.
  - Lazy read access and deferred write access to GPU memory.
  - Packing and unpacking cell's state into raw bit sequence.
  - Mixed values in expressions with variables/properties.
- Rendering.
  - Computed colors per each cell.
  - 'Moving average' color effect, for smoother visualization.
  - Projection into 2D plain from higher dimensions.
- Experiments.
  - Experiment classes describing initial state and meta-parameters.
  - Independent deterministic RNG stream per each CA.
  - 'Big Bang' and 'Primordial Soup' initial random patterns.
- Bridge with Moire GUI.
- Tests.
  - 100% code coverage by unittests.
  - Performance check script.
- Documentation.
  - Throughout API documentation via docstrings.
  - Sphinx configuration to automatically build and publish at Read
    The Docs.
  - Project overview.
  - Brief installation instructions.
  - Placeholder for tutorial.
- Miscellaneous.
  - MIT License.
  - Readme.
  - Contributing Guidelines.
  - Code of Conduct
  - Changelog.

[Unreleased]: https://github.com/a5kin/xentica/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/a5kin/xentica/compare/v0.1.0...v0.1.1
