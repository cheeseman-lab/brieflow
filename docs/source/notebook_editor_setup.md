# Working with the Notebooks

Each phase of a screen is configured in a [marimo](https://marimo.io) notebook under `analysis/`.
This page covers the day-to-day setup: an editor, running marimo on a cluster, how a notebook's values reach `config/config.yml`, and running `flow.sh`.

## Editor setup (VS Code)

Any editor works, since marimo notebooks are plain Python files and marimo serves its own editor in the browser.
VS Code is a convenient choice on a cluster because the files, terminals and forwarded ports all live on the remote machine.

1. **Connect to the cluster.** Install the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension, run *Remote-SSH: Connect to Host…* and connect to your login node.
2. **Open the screen repository**, for example with *File → Open Folder…* on `YOUR-SCREEN-REPO/`.
3. **Pick the interpreter.** Run *Python: Select Interpreter* and choose the screen's environment, `brieflow_SCREEN_NAME`. New integrated terminals then activate it, and the editor resolves `from lib....` imports to `brieflow/workflow/lib`.
4. **Recommended extensions**, installed on the remote side:

   | Extension | ID | Why |
   |---|---|---|
   | Python | `ms-python.python` | Interpreter selection, linting, go-to-definition into `lib/` |
   | Ruff | `charliermarsh.ruff` | The formatter and linter brieflow uses |
   | marimo (optional) | `marimo-team.vscode-marimo` | Opens marimo notebooks inside VS Code instead of a browser tab |

## Running a notebook

Notebooks use paths relative to `analysis/` (for example `config/config.yml`), so always start marimo from there:

```bash
cd analysis
conda activate brieflow_SCREEN_NAME
python -m marimo edit 2_sbs.py
```

brieflow pins `marimo==0.23.6`, the version the notebooks were written with.
`python -m marimo` guarantees you use the environment's copy; a bare `marimo` can pick up another install earlier on your `PATH`.
Check with `python -m marimo --version`.

### On a cluster node

The notebooks load real tiles and run segmentation, so on a shared cluster run them on a compute node rather than the login node.
Request an interactive session with your cluster's `srun` (add a GPU partition and `--gres=gpu:1` if you test the `cpsam` Cellpose model), then start marimo without a browser, listening on the node's network interface:

```bash
cd analysis
python -m marimo edit 3_phenotype.py --headless --host 0.0.0.0 --port 2718
```

marimo prints a URL with an access token.
From your laptop, forward the port through the login node and open that URL with `localhost` in place of the host name:

```bash
ssh -L 2718:<compute-node>:2718 <user>@<login-node>
```

If marimo runs on the login node instead (light notebooks only), VS Code's Remote-SSH forwards the port for you: open the *Ports* panel, or click the link marimo prints in the integrated terminal.

## How a notebook reaches config.yml

marimo notebooks are reactive: each cell is a function, the variables it returns are visible to other cells, and when a cell changes, every cell that uses its variables reruns.
In a brieflow notebook:

- The values you set live in {term}`operator parameter <Marker block>` cells, between `# === OPERATOR PARAMETERS ===` and `# === END OPERATOR PARAMETERS ===`.
  Cells marked **SET PARAMETERS** explain each one; [Notebook Parameters](notebook_parameters.md) collects the important ones.
- Test cells below them run the pipeline's own library functions on one test tile or well, so you see the effect of a value before committing to it.
- The notebook's last cell adds its section to the config (`config["sbs"] = {...}`) and rewrites `config/config.yml`.
  It depends on the parameters, so it reruns when they change and the file always holds the notebook's current values.

Two rules follow from the reactive model:

- **A variable is defined in exactly one cell.** Redefining a name in a second cell is an error; names starting with `_` are private to their cell.
- **Editing the `.py` file by hand, a new variable must appear in its cell's `return (...)` tuple and in the argument list of every cell that uses it.** The marimo editor maintains both for you; a text editor does not, and a variable missing from `return` is invisible downstream.

```{warning}
`0_preprocess.py` starts the config from scratch, keeping only the `all` and `preprocess` sections.
Every later notebook reads the existing config and replaces only its own section.
Re-running the preprocess notebook after later phases are configured therefore drops their sections; re-run those notebooks afterwards, or keep a copy of `config/config.yml`.
```

To keep a readable record of a configured notebook, `python -m marimo export html 2_sbs.py -o 2_sbs.html` runs it top to bottom and saves the output; this also rewrites its config section.

## Running flow.sh from the terminal

With the notebook configured, run its phase from the VS Code integrated terminal (or any shell) in `analysis/`:

```bash
cd analysis
bash flow.sh sbs --dry-run            # what would run
bash flow.sh sbs --backend slurm      # submit to Slurm
```

On Slurm, Snakemake itself keeps running on the login node while it submits and watches jobs, so start long runs inside `tmux` or `screen` so they survive a dropped connection.
[Running a Screen](3.running_modules.md) covers the modules and options.

## Driving it from Claude Code (optional)

The same editor can host an agent: with [brieflow-auto](brieflow_auto.md) installed in Claude Code, run `claude` in the integrated terminal from the screen repository and use `/brieflow-notebook` to configure one phase or `/brieflow-run` to run the screen.
The agent edits the same notebooks and writes the same `config/config.yml`, so you can switch between it and the manual workflow.
