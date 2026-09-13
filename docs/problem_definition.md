# RailBlock technical problem definition

## Context

RailBlock studies coordinated planning of maintenance blocks for a railway corridor. A block is a bounded period in which a section is available for defined maintenance work, subject to safe operational rules and planning constraints. The project treats the plan as decision support: a proposed plan must be reviewed by responsible railway and safety authorities before any real-world use.

The Phase 0 dataset is deliberately fictional. It is designed to exercise data validation and later scheduling research; it is not an operational model of Indian Railways.

## Inputs

The core planning problem is given:

- maintenance tasks;
- infrastructure sections;
- train movements;
- block availability;
- task deadlines and required durations;
- task criticality;
- department and resource constraints; and
- applicable maintenance dependencies.

Each task is assigned to a section and must be placed in one or more valid block windows. Future data sources may supply the equivalent information from maintenance-management and traffic/timetable systems, but no such integration is part of Phase 0.

## Decision variables

A future scheduling engine will decide whether and when each task is scheduled, the block window(s) used, grouping of compatible work, and allocation of required departmental resources. Exact mathematical variables and scheduling granularity will be selected in Phase 1 after the data contract is stable.

## Objective

Generate a feasible maintenance schedule that minimizes a weighted combination of:

- infrastructure downtime;
- train operational disruption;
- the number of separate blocks;
- unused available block capacity; and
- risk from delayed critical maintenance.

The weights must be explicit, configurable, and subject to domain review. They must not encode unverified real-railway operating assumptions.

## Feasibility constraints

Any candidate schedule must respect:

- safety constraints;
- maintenance task deadlines;
- required maintenance duration;
- section and block-window availability;
- train-movement conflicts;
- department and resource capacity constraints; and
- maintenance dependencies where applicable.

The Phase 0 validator confirms only basic data integrity. It does not certify operational feasibility, safety, dispatching rules, or real-world block authorization.

## Planning horizons and outputs

RailBlock is intended to support weekly and monthly planning. A future plan should identify scheduled tasks, assigned block windows, remaining capacity, unscheduled tasks and reasons, and explainable objective/impact measures. Day 1 contains no optimizer and emits no recommended real-world schedule.
