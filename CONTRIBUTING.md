# Contributing to Brieflow

Thank you for helping improve Brieflow. Optical pooled screens differ between labs, and we want approaches that help other users in the pipeline.

## Questions and ideas

- Ask questions and share approaches in [GitHub Discussions](https://github.com/cheeseman-lab/brieflow/discussions).
- Report a bug with the [issue template](https://github.com/cheeseman-lab/brieflow/issues/new/choose). Include your brieflow version (`git -C brieflow describe --tags` in your analysis repository) and the error.

## Pull requests

1. For a change larger than a small fix, open a Discussion or an issue first so we can agree on the approach.
2. Branch off `main` and open the pull request against `main`.
3. Keep logic in `workflow/lib`; scripts and rules stay thin. Match the style of the file you edit.
4. Run `ruff format` and `ruff check` with the repository's configuration.
5. Run the small test (`tests/small_test_analysis/run_brieflow.sh`, with `--zarr` for OME-Zarr output). It is Brieflow's test, and CI runs it on every pull request. Do not add unit-test files.
6. Update the documentation in `docs/source/` in the same pull request when the change affects what users see or set.
7. If the change adds a notebook parameter, say so in the pull request; the notebooks live in [brieflow-analysis](https://github.com/cheeseman-lab/brieflow-analysis).

The maintainers review each pull request and guide it to merge. The [development guide](https://brieflow.readthedocs.io/en/latest/5.development.html) has the details.

## Code of conduct

Everyone who takes part follows the [Code of Conduct](CODE_OF_CONDUCT.md).
