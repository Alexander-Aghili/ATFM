"""Compare seeded family generation against the pre-refactor checkout."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from atfm.sim.programs import programs_from_spec
from atfm.traces.synthetic import generate
from tests.sim.test_programs import _spec as program_spec
from tests.traces.test_synthetic import _spec as trace_spec


def sample(seed):
    trace = generate(trace_spec(seed=seed, perturb=True))
    rng = np.random.default_rng(seed)
    programs = programs_from_spec(program_spec(), rng)
    payload = trace.df.to_json(orient='split', double_precision=15) + repr(programs)
    payload += json.dumps(rng.bit_generator.state, sort_keys=True)
    return {'sha256': hashlib.sha256(payload.encode()).hexdigest(),
            'trace_rows': len(trace), 'root_programs': len(programs)}


if __name__ == '__main__':
    Path(sys.argv[1]).write_text(json.dumps({str(seed): sample(seed) for seed in range(5)}, indent=2) + '\n')
