import subprocess

discriminator_mode = 'mega' # 'medium'

if discriminator_mode == 'mega':
    model = 'large'
    p = '0.94'
elif discriminator_mode == 'medium':
    model = 'large'
    p = '0.96'

ggl = 'https://storage.googleapis.com/grover-models/'

discriminator_path = f'discrimination/generator={discriminator_mode}~discriminator=grover~discsize={discriminator_mode}~dataset=p={p}'

calls = [
    'mkdir grover/data',
    f'wget {ggl}generation_examples/generator=mega~dataset=p0.94.jsonl -P grover/data/',
    'mkdir grover/outputs',
    f'wget {ggl}{discriminator_path}/model.ckpt-1562.data-00000-of-00001 -P grover/outputs',
    f'wget {ggl}{discriminator_path}/model.ckpt-1562.index -P grover/outputs',
    f'wget {ggl}{discriminator_path}/model.ckpt-1562.meta -P grover/outputs',
    #f'wget {ggl}generation_examples/generator=mega~discriminator=grover~discsize=mega~dataset=p0.94~test-probs.npy',
    #f'wget {ggl}generation_examples/generator=mega~discriminator=grover~discsize=mega~dataset=p0.94~val-probs.npy'
]

for c in calls:
    subprocess.run(c,shell = 'bash')

import json

with open('./grover/data/generator=mega~dataset=p0.94.jsonl','r') as response:
    result = [json.loads(jline) for jline in response.read().splitlines()]

import jsonlines
with jsonlines.open('./grover/data/simple.jsonl', mode='w') as writer:
    writer.write(result[:20])