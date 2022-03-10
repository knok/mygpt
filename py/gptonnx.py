#
"""
onnx runtime test
"""

import torch
import onnxruntime
import transformers
import numpy as np

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-medium")

text = "Rust is the best programming"

inputs = tokenizer.encode_plus(text, add_special_tokens=True)
position_ids = [i for i in range(len(inputs['input_ids']))]
num_layer = 12
empty_past = []
bsize = 1
seqlen = len(inputs['input_ids'])
past_shape = [2, bsize, num_layer, 0, 64]
for i in range(num_layer):
    empty_past.append(torch.empty(past_shape).type(torch.float32).to('cpu'))

session = onnxruntime.InferenceSession("gpt2.onnx")

for inp in session.get_inputs():
    print(inp.name, inp.shape)

# for out in session.get_outputs():
#     print(out.name)

ort_inputs = {'input_ids': [inputs['input_ids']], 'attention_mask': [inputs['attention_mask']],
'position_ids': [position_ids]}
for i, past_i in enumerate(empty_past):
    ort_inputs[f'past_{i}'] = np.ascontiguousarray(
        past_i.cpu().numpy()
    )

logits = session.run(['logits'], ort_inputs)

import pprint; pprint.pprint(logits[0])
preds = logits[0][0].argmax(1)

word = tokenizer.decode(preds)

print(word)
print(inputs)
print(preds)
