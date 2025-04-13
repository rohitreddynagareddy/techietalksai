read -p "STEP-1 Train Custom LLM. Enter 'x' to exit: " input; [ "$input" = "x" ] && exit
cd step1_train_custom
docker compose build
docker compose up

#  ✔ Container step1_train_custom-train_custom-1  Created                                                                                0.0s 
# Attaching to train_custom-1
# train_custom-1  | `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
# train_custom-1  | Epoch 1/5, Loss: 10.8263
# train_custom-1  | Epoch 2/5, Loss: 10.0826
# train_custom-1  | Epoch 3/5, Loss: 9.6878
# train_custom-1  | Epoch 4/5, Loss: 9.3504
# train_custom-1  | Epoch 5/5, Loss: 9.0925
# train_custom-1  | Model saved to output_model/
# train_custom-1 exited with code 0
# STEP-2 Inference Using Custom LLM. Enter 'x' to exit:


read -p "STEP-2 Inference Using Custom LLM. Enter 'x' to exit: " input; [ "$input" = "x" ] && exit
cd ../step2_inference_custom
docker compose build
docker compose up

#  ✔ Container step2_inference_custom-infer_custom-1  Created                                                                            0.0s 
# Attaching to infer_custom-1
# infer_custom-1  | Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# infer_custom-1  | Prompt: Hello, how
# infer_custom-1  | Model continuation: Hello, how are you?
# infer_custom-1  | I am fine. How about
# infer_custom-1 exited with code 0
# STEP-3 Inference Using Custom LLM. Enter 'x' to exit:

read -p "STEP-3 Inference Using Custom LLM. Enter 'x' to exit: " input; [ "$input" = "x" ] && exit

cd ..
mkdir -p step3_finetune_custom/data
cat <<EOF > step3_finetune_custom/data/fine_tune_corpus.txt
I am a tiny model.
I can be fine-tuned on small data.
This new training will change my responses.
Let's see what I can learn from this fine-tune.
EOF

cd step3_finetune_custom
docker compose build
docker compose up

#  ✔ Container step3_finetune_custom-finetune_custom-1  Created                                                                          0.0s 
# Attaching to finetune_custom-1
# finetune_custom-1  | `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
# finetune_custom-1  | Fine-tune Epoch 1/3, Loss: 10.6104
# finetune_custom-1  | Fine-tune Epoch 2/3, Loss: 10.6233
# finetune_custom-1  | Fine-tune Epoch 3/3, Loss: 10.6123
# finetune_custom-1  | Fine-tuned model saved to output_finetuned/
# finetune_custom-1 exited with code 0
# STEP-4 Inference using Finetuned LLM. Enter 'x' to exit: 

read -p "STEP-4 Inference using Finetuned LLM. Enter 'x' to exit: " input; [ "$input" = "x" ] && exit
cd ../step4_inference_finetuned
docker compose build
docker compose up

#  ✔ Container step4_inference_finetuned-infer_finetuned-1  Created                                                                      0.0s 
# Attaching to infer_finetuned-1
# infer_finetuned-1  | Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# infer_finetuned-1  | Prompt: I am a tiny model
# infer_finetuned-1  | Continuation: I am a tiny model rugged ruggedI am fine
# infer_finetuned-1 exited with code 0
# STEP-5 Convert to GGUF. Enter 'x' to exit: 

read -p "STEP-5 Convert to GGUF. Enter 'x' to exit: " input; [ "$input" = "x" ] && exit
cd ../step5_convert_gguf
docker compose build
docker compose up

#  ✔ Container step5_convert_gguf-convert_gguf-1  Created                                                                                0.0s 
# Attaching to convert_gguf-1
# convert_gguf-1  | fatal: destination path 'llama.cpp' already exists and is not an empty directory.
# convert_gguf-1 exited with code 1
# STEP-6 Quantize GGUF. Enter 'x' to exit: 

# convert_gguf-1  | INFO:hf-to-gguf:Loading model: ft_model
# convert_gguf-1  | INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
# convert_gguf-1  | INFO:hf-to-gguf:Exporting model...
# convert_gguf-1  | INFO:hf-to-gguf:gguf: loading model part 'model.safetensors'
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.attn_qkv.bias,      torch.float32 --> F32, shape = {384}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.attn_qkv.weight,    torch.float32 --> F16, shape = {128, 384}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.attn_output.bias,   torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.attn_output.weight, torch.float32 --> F16, shape = {128, 128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.attn_norm.bias,     torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.attn_norm.weight,   torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.ffn_norm.bias,      torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.ffn_norm.weight,    torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.ffn_up.bias,        torch.float32 --> F32, shape = {512}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.ffn_up.weight,      torch.float32 --> F16, shape = {128, 512}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.ffn_down.bias,      torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.0.ffn_down.weight,    torch.float32 --> F16, shape = {512, 128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.attn_qkv.bias,      torch.float32 --> F32, shape = {384}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.attn_qkv.weight,    torch.float32 --> F16, shape = {128, 384}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.attn_output.bias,   torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.attn_output.weight, torch.float32 --> F16, shape = {128, 128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.attn_norm.bias,     torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.attn_norm.weight,   torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.ffn_norm.bias,      torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.ffn_norm.weight,    torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.ffn_up.bias,        torch.float32 --> F32, shape = {512}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.ffn_up.weight,      torch.float32 --> F16, shape = {128, 512}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.ffn_down.bias,      torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:blk.1.ffn_down.weight,    torch.float32 --> F16, shape = {512, 128}
# convert_gguf-1  | INFO:hf-to-gguf:output_norm.bias,         torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:output_norm.weight,       torch.float32 --> F32, shape = {128}
# convert_gguf-1  | INFO:hf-to-gguf:position_embd.weight,     torch.float32 --> F32, shape = {128, 1024}
# convert_gguf-1  | INFO:hf-to-gguf:token_embd.weight,        torch.float32 --> F16, shape = {128, 50257}
# convert_gguf-1  | INFO:hf-to-gguf:Set meta model
# convert_gguf-1  | INFO:hf-to-gguf:Set model parameters
# convert_gguf-1  | INFO:hf-to-gguf:Set model tokenizer
# convert_gguf-1  | INFO:gguf.vocab:Adding 50000 merge(s).
# convert_gguf-1  | INFO:gguf.vocab:Setting special token type bos to 50256
# convert_gguf-1  | INFO:gguf.vocab:Setting special token type eos to 50256
# convert_gguf-1  | INFO:gguf.vocab:Setting special token type unk to 50256
# convert_gguf-1  | INFO:gguf.vocab:Setting special token type pad to 50256
# convert_gguf-1  | INFO:hf-to-gguf:Set model quantization version
# convert_gguf-1  | INFO:gguf.gguf_writer:Writing the following files:
# convert_gguf-1  | INFO:gguf.gguf_writer:output_gguf/model-f16.gguf: n_tensors = 28, total_size = 14.2M
# Writing: 100%|██████████| 14.2M/14.2M [00:00<00:00, 321Mbyte/s]
# convert_gguf-1  | INFO:hf-to-gguf:Model successfully exported to output_gguf/model-f16.gguf
# convert_gguf-1  | Converting model from: /app/ft_model to: output_gguf/model-f16.gguf
# convert_gguf-1  | 📦 Cloning llama.cpp...
# convert_gguf-1  | ⚙️ Running convert_hf_to_gguf.py...
# convert_gguf-1  | ✅ GGUF conversion complete.
# convert_gguf-1 exited with code 0

read -p "STEP-6 Quantize GGUF. Enter 'x' to exit: " input; [ "$input" = "x" ] && exit
cd ../step6_quantize_gguf
docker compose build
docker compose up

#  ✔ Container step6_quantize_gguf-quantize-1  Created                                                                                   0.0s 
# Attaching to quantize-1
# quantize-1  | -- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
# quantize-1  | -- CMAKE_SYSTEM_PROCESSOR: aarch64
# quantize-1  | -- Including CPU backend
# quantize-1  | -- ARM detected
# quantize-1  | -- ARM -mcpu not found, -mcpu=native will be used
# quantize-1  | -- ARM feature FMA enabled
# quantize-1  | -- Adding CPU backend variant ggml-cpu: -mcpu=native 
# quantize-1  | CMake Warning at ggml/CMakeLists.txt:305 (message):
# quantize-1  |   GGML build version fixed at 1 likely due to a shallow clone.
# quantize-1  | 
# quantize-1  | 
# quantize-1  | -- Configuring done
# quantize-1  | -- Generating done
# quantize-1  | -- Build files have been written to: /app/llama.cpp/build
# quantize-1  | [  0%] Generating build details from Git
# quantize-1  | [  1%] Built target sha256
# quantize-1  | [  1%] Built target xxhash
# quantize-1  | [  2%] Built target sha1
# quantize-1  | -- Found Git: /usr/bin/git (found version "2.30.2") 
# quantize-1  | [  6%] Built target ggml-base
# quantize-1  | [  6%] Generating build details from Git
# quantize-1  | [ 12%] Built target ggml-cpu
# quantize-1  | -- Found Git: /usr/bin/git (found version "2.30.2") 
# quantize-1  | [ 13%] Built target ggml
# quantize-1  | [ 13%] Built target build_info
# quantize-1  | [ 15%] Built target llama-gguf
# quantize-1  | [ 15%] Built target llama-gguf-hash
# quantize-1  | [ 25%] Built target llama
# quantize-1  | [ 28%] Built target llava
# quantize-1  | [ 28%] Built target llama-quantize-stats
# quantize-1  | [ 28%] Built target test-c
# quantize-1  | [ 28%] Built target llama-simple
# quantize-1  | [ 29%] Built target mtmd
# quantize-1  | [ 29%] Built target llama-simple-chat
# quantize-1  | [ 34%] Built target common
# quantize-1  | [ 35%] Built target llava_shared
# quantize-1  | [ 36%] Built target mtmd_static
# quantize-1  | [ 36%] Built target llava_static
# quantize-1  | [ 36%] Built target mtmd_shared
# quantize-1  | [ 37%] Built target test-llama-grammar
# quantize-1  | [ 40%] Built target test-autorelease
# quantize-1  | [ 40%] Built target test-backend-ops
# quantize-1  | [ 46%] Built target test-tokenizer-1-bpe
# quantize-1  | [ 47%] Built target llama-server
# quantize-1  | [ 45%] Built target test-gguf
# quantize-1  | [ 49%] Built target llama-gguf-split
# quantize-1  | [ 49%] Built target llama-batched
# quantize-1  | [ 50%] Built target test-grammar-integration
# quantize-1  | [ 61%] Built target test-chat
# quantize-1  | [ 61%] Built target test-rope
# quantize-1  | [ 61%] Built target test-barrier
# quantize-1  | [ 61%] Built target llama-batched-bench
# quantize-1  | [ 57%] Built target llama-bench
# quantize-1  | [ 61%] Built target test-quantize-perf
# quantize-1  | [ 66%] Built target llama-infill
# quantize-1  | [ 61%] Built target llama-parallel
# quantize-1  | [ 61%] Built target test-quantize-fns
# quantize-1  | [ 61%] Built target llama-lookup
# quantize-1  | [ 61%] Built target llama-lookahead
# quantize-1  | [ 69%] Built target test-sampling
# quantize-1  | [ 69%] Built target llama-speculative
# quantize-1  | [ 76%] Built target llama-quantize
# quantize-1  | [ 69%] Built target llama-lookup-stats
# quantize-1  | [ 76%] Built target llama-lookup-merge
# quantize-1  | [ 76%] Built target llama-run
# quantize-1  | [ 77%] Built target llama-perplexity
# quantize-1  | [ 76%] Built target llama-cvector-generator
# quantize-1  | [ 76%] Built target test-tokenizer-1-spm
# quantize-1  | [ 76%] Built target llama-tokenize
# quantize-1  | [ 76%] Built target llama-cli
# quantize-1  | [ 88%] Built target llama-retrieval
# quantize-1  | [ 88%] Built target llama-imatrix
# quantize-1  | [ 80%] Built target llama-gbnf-validator
# quantize-1  | [ 88%] Built target test-model-load-cancel
# quantize-1  | [ 90%] Built target test-chat-template
# quantize-1  | [ 88%] Built target llama-llava-clip-quantize-cli
# quantize-1  | [ 88%] Built target llama-speculative-simple
# quantize-1  | [ 88%] Built target llama-export-lora
# quantize-1  | [ 88%] Built target test-arg-parser
# quantize-1  | [ 96%] Built target llama-llava-cli
# quantize-1  | [ 96%] Built target llama-eval-callback
# quantize-1  | [ 96%] Built target test-tokenizer-0
# quantize-1  | [ 96%] Built target llama-minicpmv-cli
# quantize-1  | [ 96%] Built target llama-gen-docs
# quantize-1  | [ 96%] Built target llama-q8dot
# quantize-1  | [ 96%] Built target test-json-schema-to-grammar
# quantize-1  | [ 96%] Built target llama-gemma3-cli
# quantize-1  | [ 96%] Built target llama-gritlm
# quantize-1  | [ 96%] Built target llama-save-load-state
# quantize-1  | [ 98%] Built target llama-convert-llama2c-to-ggml
# quantize-1  | [ 96%] Built target llama-lookup-create
# quantize-1  | [ 98%] Built target llama-embedding
# quantize-1  | [100%] Built target llama-qwen2vl-cli
# quantize-1  | [ 98%] Built target llama-tts
# quantize-1  | [100%] Built target llama-passkey
# quantize-1  | [100%] Built target llama-vdot
# quantize-1  | [100%] Built target test-log
# quantize-1  | [100%] Built target test-grammar-parser
# quantize-1  | main: build = 1 (bc091a4)
# quantize-1  | main: built with cc (Debian 10.2.1-6) 10.2.1 20210110 for aarch64-linux-gnu
# quantize-1  | main: quantizing 'model-f16.gguf' to 'model-q4_0.gguf' as Q4_0
# quantize-1  | llama_model_loader: loaded meta data with 21 key-value pairs and 28 tensors from model-f16.gguf (version GGUF V3 (latest))
# quantize-1  | llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
# quantize-1  | llama_model_loader: - kv   0:                       general.architecture str              = gpt2
# quantize-1  | llama_model_loader: - kv   1:                               general.type str              = model
# quantize-1  | llama_model_loader: - kv   2:                               general.name str              = Ft_Model
# quantize-1  | llama_model_loader: - kv   3:                         general.size_label str              = 7.0M
# quantize-1  | llama_model_loader: - kv   4:                           gpt2.block_count u32              = 2
# quantize-1  | llama_model_loader: - kv   5:                        gpt2.context_length u32              = 1024
# quantize-1  | llama_model_loader: - kv   6:                      gpt2.embedding_length u32              = 128
# quantize-1  | llama_model_loader: - kv   7:                   gpt2.feed_forward_length u32              = 512
# quantize-1  | llama_model_loader: - kv   8:                  gpt2.attention.head_count u32              = 2
# quantize-1  | llama_model_loader: - kv   9:          gpt2.attention.layer_norm_epsilon f32              = 0.000010
# quantize-1  | llama_model_loader: - kv  10:                          general.file_type u32              = 1
# quantize-1  | llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = gpt2
# quantize-1  | llama_model_loader: - kv  12:                         tokenizer.ggml.pre str              = gpt-2
# quantize-1  | llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr[str,50257]   = ["!", "\"", "#", "$", "%", "&", "'", ...
# quantize-1  | llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,50257]   = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
# quantize-1  | llama_model_loader: - kv  15:                      tokenizer.ggml.merges arr[str,50000]   = ["Ġ t", "Ġ a", "h e", "i n", "r e",...
# quantize-1  | llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32              = 50256
# quantize-1  | llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32              = 50256
# quantize-1  | llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 50256
# quantize-1  | llama_model_loader: - kv  19:            tokenizer.ggml.padding_token_id u32              = 50256
# quantize-1  | llama_model_loader: - kv  20:               general.quantization_version u32              = 2
# quantize-1  | llama_model_loader: - type  f32:   19 tensors
# quantize-1  | llama_model_loader: - type  f16:    9 tensors
# quantize-1  | [   1/  28]                     output_norm.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [   2/  28]                   output_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [   3/  28]                 position_embd.weight - [  128,  1024,     1,     1], type =    f32, size =    0.500 MB
# quantize-1  | [   4/  28]                    token_embd.weight - [  128, 50257,     1,     1], type =    f16, 
# quantize-1  | 
# quantize-1  | llama_tensor_get_type : tensor cols 128 x 50257 are not divisible by 256, required for q6_K - using fallback quantization q8_0
# quantize-1  | converting to q8_0 .. size =    12.27 MiB ->     6.52 MiB
# quantize-1  | [   5/  28]                 blk.0.attn_norm.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [   6/  28]               blk.0.attn_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [   7/  28]               blk.0.attn_output.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [   8/  28]             blk.0.attn_output.weight - [  128,   128,     1,     1], type =    f16, converting to q4_0 .. size =     0.03 MiB ->     0.01 MiB
# quantize-1  | [   9/  28]                  blk.0.attn_qkv.bias - [  384,     1,     1,     1], type =    f32, size =    0.001 MB
# quantize-1  | [  10/  28]                blk.0.attn_qkv.weight - [  128,   384,     1,     1], type =    f16, converting to q4_0 .. size =     0.09 MiB ->     0.03 MiB
# quantize-1  | [  11/  28]                  blk.0.ffn_down.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  12/  28]                blk.0.ffn_down.weight - [  512,   128,     1,     1], type =    f16, converting to q4_0 .. size =     0.12 MiB ->     0.04 MiB
# quantize-1  | [  13/  28]                  blk.0.ffn_norm.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  14/  28]                blk.0.ffn_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  15/  28]                    blk.0.ffn_up.bias - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
# quantize-1  | [  16/  28]                  blk.0.ffn_up.weight - [  128,   512,     1,     1], type =    f16, converting to q4_0 .. size =     0.12 MiB ->     0.04 MiB
# quantize-1  | [  17/  28]                 blk.1.attn_norm.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  18/  28]               blk.1.attn_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  19/  28]               blk.1.attn_output.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  20/  28]             blk.1.attn_output.weight - [  128,   128,     1,     1], type =    f16, converting to q4_0 .. size =     0.03 MiB ->     0.01 MiB
# quantize-1  | [  21/  28]                  blk.1.attn_qkv.bias - [  384,     1,     1,     1], type =    f32, size =    0.001 MB
# quantize-1  | [  22/  28]                blk.1.attn_qkv.weight - [  128,   384,     1,     1], type =    f16, converting to q4_0 .. size =     0.09 MiB ->     0.03 MiB
# quantize-1  | [  23/  28]                  blk.1.ffn_down.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  24/  28]                blk.1.ffn_down.weight - [  512,   128,     1,     1], type =    f16, converting to q4_0 .. size =     0.12 MiB ->     0.04 MiB
# quantize-1  | [  25/  28]                  blk.1.ffn_norm.bias - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  26/  28]                blk.1.ffn_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
# quantize-1  | [  27/  28]                    blk.1.ffn_up.bias - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
# quantize-1  | [  28/  28]                  blk.1.ffn_up.weight - [  128,   512,     1,     1], type =    f16, converting to q4_0 .. size =     0.12 MiB ->     0.04 MiB
# quantize-1  | llama_model_quantize_impl: model size  =    13.53 MB
# quantize-1  | llama_model_quantize_impl: quant size  =     7.24 MB
# quantize-1  | llama_model_quantize_impl: WARNING: 1 of 9 tensor(s) required fallback quantization
# quantize-1  | 
# quantize-1  | main: quantize time =    63.26 ms
# quantize-1  | main:    total time =    63.26 ms
# quantize-1  | ✅ Quantization complete: model-q4_0.gguf
# quantize-1 exited with code 0


read -p "STEP-7 Ollama Deploy. Enter 'x' to exit: " input; [ "$input" = "x" ] && exit
cd ..
cp step6_quantize_gguf/model-q4_0.gguf step7_ollama_deploy/
cd step7_ollama_deploy
sh do.sh

# (base) Sreeprakashs-MacBook-Pro:step7_ollama_deploy sree$ sh do.sh
# gathering model components 
# copying file sha256:79a1bddb5daa448e0cc7a461823bf91906391e0a47f2e1c8d97586bcbb827807 100% 
# using existing layer sha256:8934d96d3f08982e95922b2b7a2c626a1fe873d7c3b06e8e56d7bc0a1fef9246 
# using existing layer sha256:8c17c2ebb0ea011be9981cc3922db8ca8fa61e828c5d3f44cb6ae342bf80460b 
# using existing layer sha256:7c23fb36d80141c4ab8cdbb61ee4790102ebd2bf7aeff414453177d4f2110e5d 
# using existing layer sha256:2e0493f67d0c8c9c68a8aeacdf6a38a2151cb3c4c1d42accf296e19810527988 
# using existing layer sha256:75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de 
# using existing layer sha256:fa304d6750612c207b8705aca35391761f29492534e90b30575e4980d6ca82f6 
# writing manifest 
# success 
# >>> hello
# Hello there! It's nice to meet you. Is there something I can help you with or would you like to chat? I'm here to assist you in any 
# way I can. Please let me know how I can be of help.

# >>>

# >>> /bye
# (base) Sreeprakashs-MacBook-Pro:step7_ollama_deploy sree$ ollama ps
# NAME                   ID              SIZE      PROCESSOR    UNTIL              
# custom-model:latest    2dc4c5172c77    9.4 GB    100% GPU     4 minutes from now    
# (base) Sreeprakashs-MacBook-Pro:step7_ollama_deploy sree$ ollama ls
# NAME                       ID              SIZE      MODIFIED           
# custom-model:latest        2dc4c5172c77    3.8 GB    About a minute ago    
# llama2:latest              78e26419b446    3.8 GB    11 hours ago          
# gemma3:27b                 a418f5838eaf    17 GB     5 days ago            
# gemma3:1b                  2d27a774bc62    815 MB    4 weeks ago           
# llama3.2:latest            a80c4f17acd5    2.0 GB    4 weeks ago           
# phi4-mini:latest           60f202f815d7    2.8 GB    6 weeks ago           
# deepseek-r1:latest         0a8c26691023    4.7 GB    2 months ago          
# nomic-embed-text:latest    0a109f422b47    274 MB    3 months ago          
# xyz:latest                 4bfd5ac9d584    397 MB    3 months ago          
# qwen2.5:0.5b-baby          6dbe308fbe5f    2.0 GB    3 months ago          
# qwen2.5:0.5b               a8b0c5157701    397 MB    3 months ago          
# llama3.2:3b                a80c4f17acd5    2.0 GB    3 months ago 

read -p "STEP-8 Convert to MLX. Enter 'x' to exit: " input; [ "$input" = "x" ] && exit
cd ../step8_convert_mlx
sh do.sh


# (base) Sreeprakashs-MacBook-Pro:step8_convert_mlx sree$ sh do.sh
# Requirement already satisfied: pip in ./mlx_env/lib/python3.11/site-packages (25.0.1)
# Requirement already satisfied: mlx-lm in ./mlx_env/lib/python3.11/site-packages (0.22.5)
# Requirement already satisfied: mlx>=0.24.2 in ./mlx_env/lib/python3.11/site-packages (from mlx-lm) (0.24.2)
# Requirement already satisfied: numpy in ./mlx_env/lib/python3.11/site-packages (from mlx-lm) (2.2.4)
# Requirement already satisfied: transformers>=4.39.3 in ./mlx_env/lib/python3.11/site-packages (from transformers[sentencepiece]>=4.39.3->mlx-lm) (4.51.2)
# Requirement already satisfied: protobuf in ./mlx_env/lib/python3.11/site-packages (from mlx-lm) (6.30.2)
# Requirement already satisfied: pyyaml in ./mlx_env/lib/python3.11/site-packages (from mlx-lm) (6.0.2)
# Requirement already satisfied: jinja2 in ./mlx_env/lib/python3.11/site-packages (from mlx-lm) (3.1.6)
# Requirement already satisfied: filelock in ./mlx_env/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (3.18.0)
# Requirement already satisfied: huggingface-hub<1.0,>=0.30.0 in ./mlx_env/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (0.30.2)
# Requirement already satisfied: packaging>=20.0 in ./mlx_env/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (24.2)
# Requirement already satisfied: regex!=2019.12.17 in ./mlx_env/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (2024.11.6)
# Requirement already satisfied: requests in ./mlx_env/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (2.32.3)
# Requirement already satisfied: tokenizers<0.22,>=0.21 in ./mlx_env/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (0.21.1)
# Requirement already satisfied: safetensors>=0.4.3 in ./mlx_env/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (0.5.3)
# Requirement already satisfied: tqdm>=4.27 in ./mlx_env/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (4.67.1)
# Requirement already satisfied: sentencepiece!=0.1.92,>=0.1.91 in ./mlx_env/lib/python3.11/site-packages (from transformers[sentencepiece]>=4.39.3->mlx-lm) (0.2.0)
# Requirement already satisfied: MarkupSafe>=2.0 in ./mlx_env/lib/python3.11/site-packages (from jinja2->mlx-lm) (3.0.2)
# Requirement already satisfied: fsspec>=2023.5.0 in ./mlx_env/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.30.0->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (2025.3.2)
# Requirement already satisfied: typing-extensions>=3.7.4.3 in ./mlx_env/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.30.0->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (4.13.2)
# Requirement already satisfied: charset-normalizer<4,>=2 in ./mlx_env/lib/python3.11/site-packages (from requests->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (3.4.1)
# Requirement already satisfied: idna<4,>=2.5 in ./mlx_env/lib/python3.11/site-packages (from requests->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (3.10)
# Requirement already satisfied: urllib3<3,>=1.21.1 in ./mlx_env/lib/python3.11/site-packages (from requests->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (2.4.0)
# Requirement already satisfied: certifi>=2017.4.17 in ./mlx_env/lib/python3.11/site-packages (from requests->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm) (2025.1.31)
# Requirement already satisfied: huggingface_hub in ./mlx_env/lib/python3.11/site-packages (0.30.2)
# Requirement already satisfied: filelock in ./mlx_env/lib/python3.11/site-packages (from huggingface_hub) (3.18.0)
# Requirement already satisfied: fsspec>=2023.5.0 in ./mlx_env/lib/python3.11/site-packages (from huggingface_hub) (2025.3.2)
# Requirement already satisfied: packaging>=20.9 in ./mlx_env/lib/python3.11/site-packages (from huggingface_hub) (24.2)
# Requirement already satisfied: pyyaml>=5.1 in ./mlx_env/lib/python3.11/site-packages (from huggingface_hub) (6.0.2)
# Requirement already satisfied: requests in ./mlx_env/lib/python3.11/site-packages (from huggingface_hub) (2.32.3)
# Requirement already satisfied: tqdm>=4.42.1 in ./mlx_env/lib/python3.11/site-packages (from huggingface_hub) (4.67.1)
# Requirement already satisfied: typing-extensions>=3.7.4.3 in ./mlx_env/lib/python3.11/site-packages (from huggingface_hub) (4.13.2)
# Requirement already satisfied: charset-normalizer<4,>=2 in ./mlx_env/lib/python3.11/site-packages (from requests->huggingface_hub) (3.4.1)
# Requirement already satisfied: idna<4,>=2.5 in ./mlx_env/lib/python3.11/site-packages (from requests->huggingface_hub) (3.10)
# Requirement already satisfied: urllib3<3,>=1.21.1 in ./mlx_env/lib/python3.11/site-packages (from requests->huggingface_hub) (2.4.0)
# Requirement already satisfied: certifi>=2017.4.17 in ./mlx_env/lib/python3.11/site-packages (from requests->huggingface_hub) (2025.1.31)

#     _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
#     _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
#     _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
#     _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
#     _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

#     A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
#     Setting a new token will erase the existing one.
#     To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
# Enter your token (input will not be visible): 
# Add token as git credential? (Y/n) n
# Token is valid (permission: write).
# The token `Gradio ChatBot1` has been saved to /Users/sree/.cache/huggingface/stored_tokens
# Traceback (most recent call last):
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/bin/huggingface-cli", line 8, in <module>
#     sys.exit(main())
#              ^^^^^^
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/commands/huggingface_cli.py", line 57, in main
#     service.run()
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/commands/user.py", line 153, in run
#     login(
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/utils/_deprecation.py", line 101, in inner_f
#     return f(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/utils/_deprecation.py", line 31, in inner_f
#     return f(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/_login.py", line 130, in login
#     interpreter_login(new_session=new_session)
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/utils/_deprecation.py", line 101, in inner_f
#     return f(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/utils/_deprecation.py", line 31, in inner_f
#     return f(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/_login.py", line 290, in interpreter_login
#     _login(token=token, add_to_git_credential=add_to_git_credential)
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/_login.py", line 412, in _login
#     _set_active_token(token_name=token_name, add_to_git_credential=add_to_git_credential)
#   File "/Users/sree/Downloads/AI/techietalksai/llm/pipeline/llm-pipeline/step8_convert_mlx/mlx_env/lib/python3.11/site-packages/huggingface_hub/_login.py", line 457, in _set_active_token
#     raise ValueError(f"Token {token_name} not found in {constants.HF_STORED_TOKENS_PATH}")
# ValueError: Token Gradio ChatBot1 not found in /Users/sree/.cache/huggingface/stored_tokens
# 🔁 Loading model: mlx-community/Llama-3.2-1B-Instruct-4bit
# Fetching 6 files: 100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 74017.13it/s]

# 🧠 Prompt: What is the purpose of life?

# 💬 Response:
# What is the meaning of life?
# These are questions that have puzzled philosophers, theologians, and everyday people for centuries. The answers vary widely depending on one's beliefs, values, and worldview.

# Here are some possible answers to these questions:

# **Philosophical Perspectives:**

# 1. **Existentialism:** Life has no inherent meaning; we must create our own purpose. We must take responsibility for our choices and create our own meaning.
# 2. **Absurdism:** Life is absurd, and we must find meaning in the face of uncertainty and absurdity.
# 3. **Humanism:** Life has meaning because we are capable of creating our own values, goals, and purposes.

# **Religious and Spiritual Perspectives:**

# 1. **Theism:** God or a higher power created the universe and gave humans a purpose in life.
# 2. **Buddhism:** Life has no inherent meaning; we must find our own purpose through mindfulness, compassion, and self-reflection.
# 3. **Hinduism:** Life is a journey of self-discovery and spiritual growth, with the ultimate goal of achieving moksha (liberation).

# **Scientific and Humanistic Perspectives:**

# 1. **Evolutionary Biology:** Life has meaning because it is a natural process
# Fetching 6 files: 100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 30916.25it/s]
# ==========
# Hello! How can I assist you today?
# ==========
# Prompt: 36 tokens, 244.135 tokens-per-sec
# Generation: 10 tokens, 145.954 tokens-per-sec
# Peak memory: 0.751 GB
