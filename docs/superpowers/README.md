# Design records and implementation plans

This directory retains the architecture decisions and task plans used to build
ATFM. Embedded code, commands, test counts, and unchecked tasks describe the
revision at the time of writing; they are not a current installation guide or a
complete inventory of unfinished work.

- [Architecture specification](specs/2026-09-22-atfm-architecture-design.md): design decisions, hypotheses, interfaces, and later amendments.
- [L0 core forecast plan](plans/2026-09-22-l0-core-forecast-harness.md): original forecast implementation sequence.
- [L1 sidecar/proxy plan](plans/2026-09-23-l1-sidecar-proxy-traces.md): original live instrumentation and serving sequence.
- [Closed-loop simulator plan](plans/2026-09-24-l0-closed-loop-simulator.md): original simulator implementation sequence.
- [V1 completion plan](plans/2026-09-27-v1-completion.md): subsequent integration and completion tasks.

Use [implementation status](../status.md), [operations](../operations.md), and
[core development](../development/core.md) to determine current behavior. Design
intent, shipped code, unit/integration validation, and hardware evidence are
separate milestones.
