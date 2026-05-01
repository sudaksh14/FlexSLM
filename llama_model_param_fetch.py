from transformers import AutoConfig
c = AutoConfig.from_pretrained('JackFram/llama-160m')
print('model_type:         ', c.model_type)
print('hidden_act:         ', c.hidden_act)
print('intermediate_size:  ', c.intermediate_size)
print('rope_theta:         ', getattr(c, 'rope_theta', 'N/A'))
print('rms_norm_eps:       ', getattr(c, 'rms_norm_eps', 'N/A'))
print('tie_word_embeddings:', getattr(c, 'tie_word_embeddings', 'N/A'))
print('max_position_embeddings:', c.max_position_embeddings)