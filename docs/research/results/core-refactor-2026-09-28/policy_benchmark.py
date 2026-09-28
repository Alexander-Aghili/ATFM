import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pandas as pd
from atfm.proxy.config import ProxyConfig
from atfm.sim.core import Simulator
from atfm.sim.engine import EngineConfig
from atfm.sim.programs import Program, Turn
from atfm.sim.policies import NativePolicy, ProxyRulesPolicy, OraclePolicy, WorkingSetPolicy
from atfm.sim.forecast_arm import ForecastPolicy, OracleRulePolicy, fit_predictor_on_programs
from atfm.sim.kv_placement import ForecastKvPolicy, OracleKvPolicy, ForecastTouchPolicy, OracleTouchPolicy

programs = [Program(f's{i}', 'interactive' if i%3==0 else 'background','t',i*.2,
                    [Turn(1000,16,'pytest',2.+i%7,progress=[(1.,25.,100.)]),
                     Turn(200,16,'pytest',3.+i%5),Turn(200,16,None,None)])
            for i in range(120)]
engines=[EngineConfig(kv_blocks=1600,max_batch=8,prefill_tps=20000.,decode_tps=60.)]
cfg=ProxyConfig(upstream_url='unused',beta=.5)
pred,table=fit_predictor_on_programs('M2',programs,engines,np.random.default_rng(1))
factories={
 'native':lambda:NativePolicy(),
 'rules':lambda:ProxyRulesPolicy(8,cfg),
 'oracle':lambda:OraclePolicy(8,cfg),
 'working_set':lambda:WorkingSetPolicy(1400,window=8),
 'forecast':lambda:ForecastPolicy(8,cfg,pred,table,[5.,20.],n=32),
 'oracle_rule':lambda:OracleRulePolicy(8,cfg,horizons=[5.,20.],n=8),
 'forecast_kv':lambda:ForecastKvPolicy(8,cfg,pred,table,[5.,20.],n=32),
 'oracle_kv':lambda:OracleKvPolicy(8,cfg),
 'forecast_touch':lambda:ForecastTouchPolicy(8,cfg,pred,table,[5.,20.],n=32),
 'oracle_touch':lambda:OracleTouchPolicy(8,cfg),
}
out={}
for name,factory in factories.items():
 times=[]; hashes=[]
 for repeat in range(4):
  rng=np.random.default_rng(7)
  sim=Simulator(programs,engines,factory(),tick_s=5,max_hold_s=5,rng=rng)
  start=time.perf_counter();rows=sim.run();elapsed=time.perf_counter()-start
  digest=hashlib.sha256(pd.util.hash_pandas_object(rows,index=True).values.tobytes()
                       +json.dumps(rng.bit_generator.state,sort_keys=True).encode()
                       +json.dumps(sim.session_log,sort_keys=True).encode()).hexdigest()
  if repeat:times.append(elapsed);hashes.append(digest)
 assert len(set(hashes))==1,name
 out[name]={'seconds':times,'median_s':statistics.median(times),'sha256':hashes[0],'rows':len(rows)}
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
print({k:round(v['median_s'],4) for k,v in out.items()})
