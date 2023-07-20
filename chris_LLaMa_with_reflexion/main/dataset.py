from reflexion.programming_runs.utils import read_jsonl, read_jsonl_gz
data = read_jsonl("../../reflexion/programming_runs/benchmarks/humaneval-py_hardest50.jsonl")
print(data)
