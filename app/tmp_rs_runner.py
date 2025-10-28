# temp runner to import RelationScorer without evaluating top-level get_config
import importlib, types, sys

# load module source and exec it with a fake get_config to avoid config errors
from pathlib import Path
p = Path(__file__).with_name('relation_scorer.py')
src = p.read_text()

# Create stub modules to satisfy top-level imports in relation_scorer
stub_config_mod = types.ModuleType('config')
def _get_config():
    stub = types.SimpleNamespace()
    stub.scorer = types.SimpleNamespace(backend='rule', use_ema=True, decay_factor=1.5)
    stub.ollama = types.SimpleNamespace(model='ollama-test')
    stub.llm = types.SimpleNamespace(provider='none')
    stub.env = types.SimpleNamespace(debug=False)
    return stub
stub_config_mod.get_config = _get_config
sys.modules['config'] = stub_config_mod

stub_azure = types.ModuleType('azure_clients')
def fake_get_azure_chat_completion_client(cfg):
    return (None, None)
stub_azure.get_azure_chat_completion_client = fake_get_azure_chat_completion_client
sys.modules['azure_clients'] = stub_azure

# execute relation_scorer source in a fresh globals dict
g = {'__name__': 'relation_scorer_tmp'}
exec(src, g)
RelationScorer = g['RelationScorer']

if __name__ == '__main__':
    rs = RelationScorer(backend='rule', use_ema=True, verbose=True)
    logs=[{'speaker':'A','utterance':'こんにちは'},{'speaker':'B','utterance':'そうだね'},{'speaker':'A','utterance':'いいね'}]
    out, trace = rs.get_scores(logs,['A','B'],return_trace=True,update_state=True)
    print('OUT:', out)
    print('\nTRACE:')
    for t in trace:
        print(t)
