# RailBlock technical problem definition

## Context and current scope

RailBlock studies coordinated planning of maintenance blocks for a railway
corridor. A block is a bounded period in which a section is available for
defined maintenance work, subject to the constraints modeled by the prototype.
The project treats a schedule as decision support: responsible railway and
safety authorities would need to review any real-world plan.

The synthetic-data foundation was completed in Phase 0. The current
implementation is a Phase 2.4.7 risk-aware scheduling prototype, with a
dashboard that displays its outputs. The data and results are fictional and
are not an operational model of Indian Railways.

## Inputs

The implemented experiment reads:

- maintenance tasks, including required duration, earliest start, deadline,
  criticality, section, and other task attributes;
- infrastructure sections and stations;
- available maintenance block windows;
- train movements, including service date, entry time, exit time, and section;
- synthetic asset features and maintenance history for risk modelling; and
- generated asset-risk scores for the risk-aware optimizer.

`src.data.validation` checks the required CSV structure and cross-file
references before pipeline stages use the data.

## Implemented decision variables

Candidate generation first creates every task/block placement that fits the
task's basic timing and section eligibility rules. The CP-SAT model then makes
the following decisions:

- a Boolean assignment variable for each feasible candidate placement;
- an optional maintenance interval associated with that candidate; and
- a Boolean variable indicating whether an available block is used.

A task may be deliberately left unscheduled because the model uses an
at-most-one constraint, rather than requiring every task to be assigned.

## Implemented objective

The baseline model maximizes a weighted score for scheduled work, rewarding
criticality and urgency while penalizing unnecessary use of maintenance blocks.
The risk-aware model preserves the feasibility model and adds asset-risk-based
priority. Its risk weights can be varied by the sensitivity experiment.

The objective is a project-specific priority function. An `OPTIMAL` solver
status proves optimality only for that encoded function and the modeled
constraints; it does not prove that a plan is operationally safe or globally
best for a real railway.

## Implemented feasibility constraints

The current models enforce:

- at most one selected candidate for each maintenance task;
- placement only in candidate windows generated as feasible;
- a selected interval fitting completely inside its block;
- task earliest-start and deadline timing;
- no overlapping selected maintenance intervals in the same block; and
- no overlap between selected maintenance and modeled train movements in the
  same section.

The schedule validator independently rechecks selected placements, including
task/block timing and train conflicts. This provides a useful regression check
but is not a substitute for real-world safety validation.

## Not yet modeled

The following are important future capabilities, not current guarantees:

- department/resource capacity constraints;
- maintenance dependency constraints;
- real dispatching, possession, safety, and authorization rules;
- live maintenance-management or timetable integrations;
- a production API, persistence layer, or approval workflow; and
- dashboard-driven re-optimization for What-if scenarios.

## Outputs and interpretation

The solver writes schedule CSV outputs and the project can produce validation,
explanation, competition, and risk-weight sensitivity analyses. These outputs
should be interpreted as reproducible synthetic experiments. They identify
what the current model chooses under its explicit assumptions, not what should
be deployed on a real railway corridor.
