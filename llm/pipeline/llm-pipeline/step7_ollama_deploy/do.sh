


cp ../step6_quantize_gguf/model-q4_0.gguf .


# ollama create tinyllama-q4

ollama create custom-model -f Modelfile

ollama run custom-model