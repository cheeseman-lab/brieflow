# Running brieflow with an Agent (brieflow-auto)

[brieflow-auto](https://github.com/cheeseman-lab/brieflow-auto) is a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) plugin that orchestrates a whole brieflow screen through an agent, from setup to the last phase.
It drives the same marimo notebooks, `config/config.yml` and `flow.sh` described in these docs; it adds the bookkeeping, checks and recovery a person would otherwise do by hand.
Using it is optional, and a screen can move between it and the manual workflow at any point.

```{note}
The brieflow-auto repository is currently private; ask the Cheeseman lab for access.
```

## What it does

1. **You describe the screen once.** `/brieflow-setup` provisions the host (optionally a cloud VM), stages the raw data, clones the screen repository, builds the conda environment and helps you write {term}`screen.yaml`.
2. **You answer a short interview.** Every notebook {term}`operator parameter <Marker block>` is resolved from one source: `screen.yaml`, a lab default, a probe of the raw data (file patterns, channel order, tile counts), an agent-tuned sweep, or a question to you. A value left blank in `screen.yaml` is asked, never guessed. Answers and their provenance are recorded in `.brieflow/interview.json`.
3. **You approve a launch review.** Before any compute is spent, the runner writes `.brieflow/launch_review.html`: the phases in range, every active parameter grouped by where its value came from, remaining gaps, the disk needed per phase, and the compute target. The run starts only when you re-run with `--approved` against the review you read.
4. **The agent runs each phase.** For every phase in range it configures the notebook in a live marimo kernel (sweeping parameters on a test tile where the notebook calls for tuning), runs a static preflight check of the config against brieflow's own rule code, submits the module and monitors it, recovers from classified failures (more memory after an out-of-memory kill, a higher open-file limit, a wait after filesystem latency), and gates the phase on its `eval/` QC outputs before moving on.
5. **Everything is recorded.** Beside the outputs, `.brieflow/` keeps the interview, the launch review, a run manifest (plugin, brieflow and screen repository versions, and the flags used), a decision journal of judgment calls, and a findings log. Each configured notebook is exported to HTML.
6. **Pipeline defects become upstream issues.** When a run stops on something that is a defect in brieflow rather than in the run, the agent investigates it and opens an issue on cheeseman-lab/brieflow with the evidence, a minimal backward-compatible fix proposal and regression tests (`/brieflow-issue`). It never changes pipeline code during a run.

The run pauses only on a real blocker: an unanswered parameter, a preflight failure, a failed QC gate, a failure that recovery cannot fix, or a notebook error.
Fix the cause and re-run the same command; finished phases are skipped.

## Install and start

In Claude Code:

```text
/plugin marketplace add cheeseman-lab/brieflow-auto
/plugin install brieflow-auto@brieflow-auto
/reload-plugins
```

Then, on the machine that holds the data (a cluster login node, a workstation or a cloud VM), start Claude Code in the screen repository and run:

| Command | Use |
|---|---|
| `/brieflow-setup` | First-time onboarding: host, raw data, screen repository, environment and `screen.yaml` |
| `/brieflow-notebook` | Configure and run one phase notebook interactively, and export its HTML |
| `/brieflow-run` | Run the screen through the per-phase cycle, for all phases or a range (`--only`, `--from`, `--through`) |
| `/brieflow-status` | Read-only status of jobs, progress and recent failures |
| `/brieflow-qc` | Gate one phase on its QC outputs |
| `/brieflow-issue` | Open a brieflow defect report |

The [brieflow-auto README](https://github.com/cheeseman-lab/brieflow-auto) lists every command and describes updating and uninstalling the plugin.

## How it fits with the manual workflow

- The plugin writes the same `config/config.yml` you would, through the same notebooks, so you can open any notebook and see or change what was set.
- `flow.sh` detects an installed brieflow-auto and uses its resource resolver to pass per-rule memory limits to Snakemake (see [Running on Slurm](3.running_modules.md#running-on-slurm)).
- The agent runs where the data and environment are; it is not a remote dispatcher.
