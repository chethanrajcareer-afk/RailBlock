# Architecture

## Current Phase 0 architecture

```text
Synthetic CSV data
       |
       v
Data validation
       |
       v
Clean pandas DataFrames
       |
       v
Scheduling / optimization engine (reserved, not implemented)
       |
       v
Schedule evaluation (reserved, not implemented)
```

`src.data.generate_synthetic_data` creates deterministic fictional inputs. `src.data.validation` owns the data contract: Pydantic row schemas, controlled vocabularies, time checks, and cross-file reference checks. It returns cleaned pandas DataFrames only after validation succeeds.

The `optimization`, `prediction`, and `api` packages are intentional placeholders. Keeping them separate prevents synthetic-data generation and validation from becoming coupled to later planning, ML, or application concerns.

## Intended future architecture

```text
Maintenance systems: TMS / SMMS / TDMS
                  |
                  v
            Data integration
                  |
                  v
      Maintenance priority engine  <--- Traffic / timetable data
                  |                           |
                  +-------------+-------------+
                                |
                                v
                    Optimization engine
                                |
                                v
                  Weekly / monthly block plan
                                |
                                v
                               API
                                |
                                v
               Dashboard / control-room interface
```

This diagram describes a research direction, not an integration commitment. Any introduction of live data must add source-specific contracts, access controls, provenance, privacy/security review, and validation before it can influence a plan.

## Design boundaries

- Data generation remains independent of optimization logic.
- Validation is shared infrastructure, not business logic embedded in a scheduler.
- Future prediction/ML code remains in `src/prediction/` and is optional; it must not be presented as a substitute for validated operational constraints.
- APIs and dashboards consume documented scheduling outputs rather than owning scheduling rules.
- SQLite or another persistence layer will be considered only once CSV workflows no longer meet the project’s needs.
