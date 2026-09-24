import sys

import yaml

from atfm.collect.driver import run_collection
from atfm.collect.jobs import CollectionSpec

if __name__ == "__main__":
    spec = CollectionSpec(**yaml.safe_load(open(sys.argv[1])))
    print(run_collection(spec))
