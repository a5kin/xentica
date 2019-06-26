# Changelog
All notable changes to Xentica project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Core.
  - Disable ``nvcc`` compiler warnings.
- Exceptions.
  - Raise exception when assigning to ``DeferredExpression``.
  - Raise exception if ``Variable`` has no initial value.
- QA.
  - Disable faulty ``pylint`` issues.
  - Raise max locals limit for ``pylint``.
  - Remove PyPy3 from supported versions.
- Refactoring.
  - Change ``Constant`` API to rid of eval.
  - Move ``DeferredExpression`` to separate module.
  - Move code generation to ``CellularAutomaton.__init__``
  - Change ``Constant`` defining mechanism.

### Added

- Core.
  - Add ``TotalisticRuleProperty``.
  - Add ``RandomProperty``.
  - Add ``FloatVariable``.
  - Add ``long int`` to supported types.
  - Add ``BscaDetectorMixin`` to ``Property``.
  - Add ``Parameter`` class.
  - Integrate parameters to core engine.
  - Add meta-params population from ``Experiment`` class.
  - Implement interactive parameters.
  - Add validation for field's dimensionality.
- Seeds.
  - Add ``ChainedPattern`` functionality.
  - Add support for ``RandInt`` in mixed expressions.
  - Add option to ``RandInt`` making it constant.
- Tools.
  - Add ``xentica.tools`` package.
  - Add color conversion helpers module (``tools.color``).
    - Basic HSV to RGB conversion.
	- Genome coloring: positional and modular methods.
  - Add math functions wrappers (``tools.xmath``)
    - ``Min`` / ``max`` functions over a number of arguments.
	- ``Float`` / ``int`` type casting.
	- Counting a number of non-zero bits in integer (``popc``).
  - Add CA rules helpers (``tools.rules``).
    - Convert Golly nonation to integer.
	- Convert integer to Golly notation.
  - Add gennome manipulation helpers (``tools.genetics``).
    - Crossover several genomes in stochastic way.
	- Mutation during genomes' crossover.
- Examples.
  - Add ``EvoLife`` example model.
  - Add ``NoiseTV`` example model.
- Tests.
  - Implement full test suite with ``tox``.
  - Add test for broad bit width.
  - Add test for ``RendererPlain`` class.
  - Add test for illegal assign to ``DeferredExpression``.
  - Add test for ``Variable`` without init value.
- Documentation.
  - Add separate section for testing.
  - Add core.expressions to docs.
- Miscellaneous.
  - Add ``NoiseTV`` and ``EvoLife`` to benchmark.

### Fixed
- Major fixes.
  - Fix augmented assigns to properties and variables.
  - Fix direct assignments to variables.
  - Fix variables names in declarations.
  - Fix variable's fallback name.
  - Fix ``BscaDetectorMixin``.
  - Fix ``ColorEffect``.
  - Fix ``dtype``/``ctype`` usage in ``CellularAutomaton`` class.
- Minor fixes.
  - Remove unnecessary conditions.
  - Fix default field size.
  - Fix minor codestyle issues.
  - Fix typos and grammar in docs.
  - Fix imports in examples.

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
