#!/usr/bin/env python3
"""Download Qwen2.5-3B model with SSL workarounds."""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests session
import requests
old_request = requests.Session.request
def new_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, *args, **kwargs)
requests.Session.request = new_request

# Patch httpx
import httpx
original_init = httpx.Client.__init__
def patched_init(self, *args, **kwargs):
    kwargs.setdefault('verify', False)
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_init

from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

QWEN_PATH = 'models/qwen2.5-3b'
Path(QWEN_PATH).mkdir(parents=True, exist_ok=True)

print('='*60)
print('Downloading Qwen2.5-3B (3 Billion parameters)')
print('='*60)

print('\n[1/2] Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
tokenizer.save_pretrained(QWEN_PATH)
print('✓ Tokenizer saved!')

print('\n[2/2] Downloading model (this may take 10-15 minutes)...')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B', 
    trust_remote_code=True, 
    low_cpu_mem_usage=True
)
model.save_pretrained(QWEN_PATH)
print('✓ Model saved!')

print('\n' + '='*60)
print(f'✅ Qwen2.5-3B downloaded to: {QWEN_PATH}')
print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
print('='*60)
