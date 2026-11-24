# Architecture

## Current implementation: Phase 2.4.7, with a partial Phase 2.5 dashboard

RailBlock is an academic decision-support prototype. It uses deterministic
synthetic data to demonstrate a complete pipeline from data validation to a
validated maintenance schedule. It is not connected to live railway systems
and must not be used to authorize real maintenance work.

```text
Synthetic CSV inputs
       |
       v
Shared data validation ----------------------------------------+
       |                                                       |
       v                                                       v
Feature / maintenance-history generation                Task and block inputs
       |                                                       |
       v                                                       v
Risk dataset -> Random Forest -> asset-risk scores      Candidate generation
                       |                                       |
                       +---------------+-----------------------+
                                       |
                                       v
                         Baseline or risk-aware CP-SAT model
                                       |
                                       v
                         Schedule CSV + validation report
                                       |
                                       v
              Sensitivity, competition, and explanation analyses
                                       |
                                       v
                    Streamlit decision-support dashboard
```

### Module responsibilities

- `src/data/` creates the deterministic synthetic inputs, builds asset
  features and maintenance history, and validates the shared CSV contract.
- `src/prediction/` builds a risk dataset, trains baseline and Random Forest
  models, and writes asset-risk scores. These models are demonstrations on
  synthetic labels, not production forecasts.
- `src/optimization/` creates feasible task/block candidates, builds baseline
  and risk-aware OR-Tools CP-SAT models, solves them, validates schedules, and
  produces analysis outputs.
- `dashboard/` reads validated synthetic inputs and generated outputs to show
  corridor, task, block, train, and risk views. Its interactive What-if page
  is still a placeholder; the dashboard does not run a new optimization model
  from user-entered changes.
- `src/api/` remains intentionally reserved; the project has no API service.

The implemented scheduling model uses optional intervals and binary candidate
assignment decisions. It enforces at-most-one assignment per task, placement
inside an eligible block, non-overlap within a block, and train-movement
separation. The risk-aware model retains those constraints while changing the
priority score used by the objective.

## Current boundaries

- Inputs, model outputs, and reported results are synthetic and deterministic.
- Data validation checks the input contract; schedule validation checks the
  modeled schedule. Neither is a railway safety, dispatching, or authorization
  system.
- Department/resource capacity constraints and maintenance-dependency logic
  are represented in the problem domain but are not yet modeled as CP-SAT
  constraints.
- The generated CSV files are experiment outputs, not a persistent planning
  database or an operational integration.

## Planned direction

```text
Validated live data sources
           |
           v
Source-specific contracts, provenance, and access controls
           |
           v
Operationally reviewed prediction and optimization services
           |
           v
Approved planning workflow, API, and dashboard
```

This is a research direction, not an integration commitment. Any live-data or
operational use would require domain review, source-specific validation,
security and privacy controls, auditability, and explicit safety governance.
