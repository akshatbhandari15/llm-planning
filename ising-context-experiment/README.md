# Ising-Style Context Experiment

This standalone experiment probes how an autoregressive language model changes
its next-token distribution and attention structure when context is increased
and then decreased.

The design is inspired by Ising-model observables:

- **Magnetization analogue**: target-vs-competitor logit margin.
- **Susceptibility analogue**: discrete derivative of magnetization wrt context.
- **Entropy analogue**: prediction entropy of next-token distribution.
- **Hysteresis**: forward vs backward sweep mismatch area.

## Folder structure

```text
ising-context-experiment/
  run_experiment.py
  requirements.txt
  README.md
  ising_context_experiment/
    __init__.py
    prompts.py
    model_probe.py
    metrics.py
    experiment.py
    plots.py
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py --model gpt2 --device cpu --output-dir ./results_ising
```

You can control resolution and sweep ranges:

```bash
python run_experiment.py --min-context 8 --max-context 160 --n-steps 7 --max-prompts 6
```

## Outputs

The run creates:

- `observations.csv`: per-prompt/per-direction/per-context measurements.
- `summary.json`: aggregate statistics and hysteresis metrics.
- Plot files in `plots/`:
  - `magnetization_curves.png`
  - `entropy_curves.png`
  - `susceptibility_curves.png`
  - `hysteresis_scatter.png`
