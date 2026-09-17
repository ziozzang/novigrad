# FunctionGemma–novi domain adapter

Experimental LoRA adapter for four typed tools. Base: google/functiongemma-270m-it, original BF16; rank 8, alpha 16, Q/V projections, 368,640 trainable parameters. Trained on 76 handcrafted English/Korean examples for 80 steps, batch 2, seed 731.

Fresh exact-call accuracy: 13/16 versus base 6/16; explicit development 8/8; semantic rejection 2/4. One small, author-created evaluation and one training seed; three Korean failures remain. Not a general-purpose or production control model. See examples/function_bridge/README.md and adjacent raw results for methodology and limitations.

Weights are saved locally as adapter_model.safetensors and excluded from Git. Recreate with examples/function_bridge/finetune.py. Applicable Gemma model terms remain in force; the repository MIT license does not relicense model weights.
