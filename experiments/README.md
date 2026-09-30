# experiments/

Records of experiments. **No experiments have been run yet** (Phase 0).

Generated artefacts (checkpoints, logs, figures, metric files) go to
`outputs/`; this directory holds the human-readable record of what was run,
why, and what was found.

## Record template

Create one Markdown file per experiment, named `<experiment_id>.md`
(IDs from `seismic_ml.utils.reproducibility.generate_experiment_id`):

```markdown
# <experiment_id>

- Phase:
- Hypothesis / question:          (e.g. H1, see PROJECT_CONTEXT.md §18)
- Model / baseline:               (see PROJECT_CONTEXT.md §7)
- Dataset + version:
- Split strategy:                 (must satisfy PROJECT_CONTEXT.md §12)
- Config snapshot:                outputs/results/<experiment_id>/config.yaml
- Seeds:
- Software versions:              (get_environment_info())
- Git commit:
- Results location:               outputs/results/<experiment_id>/

## Results
(only numbers actually produced by this run; mean ± spread over seeds)

## Interpretation and caveats

## Follow-up / decision
(link Decision Log entries if this experiment changed a decision)
```

Rules: never edit results after the fact; report negative results;
never report numbers that were not produced by a recorded run.
