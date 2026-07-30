"""
Prints the architecture fields needed to build a matching FlexLLaMAConfig top
level for a given HF causal-LM checkpoint (hidden size, layers, heads, kv heads,
intermediate size, vocab size, rope/rmsnorm settings). Used when adding a new
entry to TEACHER_PRESETS in config/experiments.py.
"""
import sys
from transformers import AutoConfig

for name in sys.argv[1:]:
    c = AutoConfig.from_pretrained(name)
    print(f"=== {name} ===")
    print('model_type:             ', c.model_type)
    print('hidden_size:             ', c.hidden_size)
    print('num_hidden_layers:       ', c.num_hidden_layers)
    print('num_attention_heads:     ', c.num_attention_heads)
    print('num_key_value_heads:     ', getattr(c, 'num_key_value_heads', c.num_attention_heads))
    print('intermediate_size:       ', c.intermediate_size)
    print('vocab_size:              ', c.vocab_size)
    print('hidden_act:              ', c.hidden_act)
    print('rope_theta:              ', getattr(c, 'rope_theta', 'N/A'))
    print('rms_norm_eps:            ', getattr(c, 'rms_norm_eps', 'N/A'))
    print('tie_word_embeddings:     ', getattr(c, 'tie_word_embeddings', 'N/A'))
    print('max_position_embeddings: ', c.max_position_embeddings)
    print()
