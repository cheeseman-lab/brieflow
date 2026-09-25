# Brieflow

[![Release](https://img.shields.io/github/v/release/cheeseman-lab/brieflow)](https://github.com/cheeseman-lab/brieflow/releases)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-brieflow.readthedocs.io-brightgreen)](https://brieflow.readthedocs.io)
[![Tests](https://github.com/cheeseman-lab/brieflow/actions/workflows/test_analysis.yml/badge.svg)](https://github.com/cheeseman-lab/brieflow/actions/workflows/test_analysis.yml)
[![Discord](https://img.shields.io/badge/forum-discord-7289da)](https://discord.gg/yrEh6GP8JJ)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Nature Communications](https://img.shields.io/badge/Nat%20Commun-10.1038%2Fs41467--026--73643--7-b31b1b)](https://doi.org/10.1038/s41467-026-73643-7)

![Brieflow pipeline](images/brieflow_info.png)

Brieflow is an extensible computational pipeline for high-throughput analysis of optical pooled screening data.

This repo contains the source code for running an OPS screen analysis.
[Brieflow Analysis](https://github.com/cheeseman-lab/brieflow-analysis) contains configuration notebooks/files and execution scripts for running an OPS screen analysis.

## Getting Started

We strongly suggest that Brieflow is set up with the companion [Brieflow Analysis](https://github.com/cheeseman-lab/brieflow-analysis) repository.

Full details on setup, installation, test data, usage, module details, and contribution guides:  
**https://brieflow.readthedocs.io**

Each phase is configured in a [marimo](https://marimo.io) notebook in brieflow-analysis and run with `flow.sh`; the output can be written as TIFF or OME-Zarr.
The whole pipeline can also be driven by an agent with [brieflow-auto](https://github.com/cheeseman-lab/brieflow-auto), a Claude Code plugin.

## Citing Brieflow

Brieflow was created by [Matteo Di Bernardo](https://github.com/mat10d), [Roshan Kern](https://github.com/roshankern), and others in the [Cheeseman Lab](https://cheesemanlab.wi.mit.edu/).
Brieflow was started in 2024 and is actively being developed.
If you are interested in contributing please reach out!

If you use our code please cite this manuscript:

```
@article{dibernardo2026brieflow,
  title={Brieflow: an integrated computational pipeline for high-throughput analysis of optical pooled screening data},
  author={Di Bernardo, Matteo and Kern, Roshan S. and Cepeda Diaz, Ana Karla and Mallar, Alexa and Choi, Samuel J. and Nutter-Upham, Andrew and Lourido, Sebastian and Blainey, Paul C. and Cheeseman, Iain M.},
  journal={Nature Communications},
  volume={17},
  pages={6997},
  year={2026},
  doi={10.1038/s41467-026-73643-7}
}
```

## Contributing

We welcome community contributions to Brieflow. Optical pooled screens vary between labs and we would love to add and share approaches that you have taken to your data such that the community can make use of this!

Feel free to:
- Give the repo a star to boost Brieflow's visibility!
- Join Brieflow's [Discord](https://discord.gg/yrEh6GP8JJ) to chat with the developers.
- File a [GitHub issue](https://github.com/cheeseman-lab/brieflow/issues) to share comments and issues.
- Clone the repository, create a new branch, and submit a [pull request](https://github.com/cheeseman-lab/brieflow/compare) as detailed in the [pull request template](.github/pull_request_template.md).

Make sure to review the Brieflow [development guide](https://brieflow.readthedocs.io/en/latest/5.development.html) to understand how to best contribute!

