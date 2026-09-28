# Function-boundary refactor

This pass applies a maximum of 20 physical lines per function, measured from
`def` through the last statement (including multiline signatures, docstrings,
comments, and blank lines). Helpers represent named operations; splitting must
not hide work in long expressions, introduce recursive orchestration, or change
algorithmic complexity. Existing mutable state, event ordering, RNG ordering,
public entry points, and failure behavior remain part of the contract.

## Runtime boundaries

- Proxy creation registers methods on a per-app runtime. Prediction fallback,
  session bookkeeping, admission, response forwarding, and stream cleanup are
  separate operations. State remains available through `app.state`.
- Board creation uses a per-app runtime; scraping and controller planning have
  independent boundaries. Directive results remain cached by snapshot identity.
- ControlLoop separates hold validation, bounded batch delivery, touch delivery,
  tier actions, and accounting. Batch acknowledgements are still validated.
- Subprocess capture separates launch, byte collection, process-group timeout
  cleanup, and completion. A shared parser event dispatcher handles both live
  subprocess output and completed executor output without changing event order.

Initial runtime stage: 217 tests passed across proxy, board, control, and sidecar;
one dependency deprecation warning. Broader validation follows subsequent stages.
