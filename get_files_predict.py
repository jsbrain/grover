import subprocess
import json
import jsonlines
import os

import argparse

parser = argparse.ArgumentParser(description='Contextual generation (aka given some metadata we will generate articles')
parser.add_argument(
    '--model_size',
    default='large',
    type=str,
    help='Model size can use mega or large, default large',
)
parser.add_argument(
    '--use_samples',
    default=20,
    type=int,
    help='Number of samples to use',
)

args = parser.parse_args()
model_size = args.model_size
use_samples = args.use_samples

if model_size == 'mega':
    discriminator_mode = 'mega'
    p = '0.94'
elif model_size == 'large':
    discriminator_mode = 'medium'
    p = '0.96'

use_split = 'val'

ggl = 'https://storage.googleapis.com/grover-models/'

discriminator_path = f'discrimination/generator={discriminator_mode}~discriminator=grover~discsize={discriminator_mode}~dataset=p={p}'

calls = [
    'mkdir grover/data',
    f'wget {ggl}generation_examples/generator=mega~dataset=p0.94.jsonl -P ./data/',
    'mkdir grover/outputs',
    f'wget {ggl}{discriminator_path}/model.ckpt-1562.data-00000-of-00001 -P ./outputs',
    f'wget {ggl}{discriminator_path}/model.ckpt-1562.index -P ./outputs',
    f'wget {ggl}{discriminator_path}/model.ckpt-1562.meta -P ./outputs',
    #f'wget {ggl}generation_examples/generator=mega~discriminator=grover~discsize=mega~dataset=p0.94~test-probs.npy',
    #f'wget {ggl}generation_examples/generator=mega~discriminator=grover~discsize=mega~dataset=p0.94~val-probs.npy'
]

for c in calls:
    subprocess.run(c,shell = 'bash')

with open('./data/generator=mega~dataset=p0.94.jsonl','r') as response:
    result = [json.loads(jline) for jline in response.read().splitlines()]

for r in result[:use_samples]:
    r['split'] = use_split

f = jsonlines.open('./data/simple.jsonl', mode='a')
for r in result[:use_samples]:
    f.write(r)
f.close()