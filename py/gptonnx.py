#
"""
onnx runtime test
"""

import onnxruntime
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")

text = "Rust is the best programming [MASK]."

inputs = tokenizer.encode_plus(text, add_special_tokens=True)

session = onnxruntime.InferenceSession("./distilgpt2.onnx/model.onnx")

for inp in session.get_inputs():
    print(inp.name, inp.shape)

for out in session.get_outputs():
    print(out.name)

logits = session.run(['logits'], 
{'input_ids': [inputs['input_ids']], 'attention_mask': [inputs['attention_mask']]})

import pprint; pprint.pprint(logits[0])
preds = logits[0][0].argmax(1)

word = tokenizer.decode(preds)

print(word)
