conda activate vllm

python -m vllm.entrypoints.openai.api_server \
    --model path/to/your/Qwen3-VL-8B-Instruct \
    --served-model-name qwen3-vl-plus \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192