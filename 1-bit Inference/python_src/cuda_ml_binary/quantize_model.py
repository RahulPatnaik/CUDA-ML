import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from cuda_ml_binary.modules import BinaryLinear, BinaryActivation, calculate_alpha

def quantize_linear_layer(linear_layer: nn.Linear):
    """
    Converts a standard nn.Linear layer to a BinaryLinear layer.
    """
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    has_bias = linear_layer.bias is not None

    binary_layer = BinaryLinear(in_features, out_features, bias=has_bias)

    # Transfer quantized weights and alpha
    with torch.no_grad():
        binary_layer._quantize_weights(linear_layer.weight)
        if has_bias:
            binary_layer.bias.copy_(linear_layer.bias)

    return binary_layer

def quantize_transformer_model(model: nn.Module, device: torch.device):
    """
    Quantizes a Hugging Face Transformer model's Linear layers to 1-bit.
    For simplicity, this will iterate through modules and replace nn.Linear.
    For full BNN, you'd also need to insert BinaryActivation layers appropriately.
    """
    print(f"Quantizing model: {model.__class__.__name__}")
    quantized_model = model.to(device)

    # A recursive function to find and replace modules
    def replace_modules(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(f"  - Quantizing Linear layer: {name} (in_features={child.in_features}, out_features={child.out_features})")
                setattr(module, name, quantize_linear_layer(child).to(device))
                # For full BNN, you would also insert a BinaryActivation after this layer
                # This often depends on the specific architecture of the Transformer block
                # For example, after the Linear layer in the FFN and after attention output.
                # setattr(module, f"{name}_act", BinaryActivation().to(device)) # Example insertion

            else:
                replace_modules(child) # Recurse into submodules
        return module

    # Start replacement from the top level of the model
    quantized_model = replace_modules(quantized_model)
    print("Model quantization complete.")
    return quantized_model

if __name__ == "__main__":
    # Example usage for DistilBERT
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move original model to device to ensure weights are on GPU if that's preferred
    model.to(device)

    # Quantize the model
    quantized_model = quantize_transformer_model(model, device)

    # print("\nOriginal Model Structure (partial):")
    # print(model.distilbert.transformer.layer[0].sa_proj) # Self-attention projection
    # print("\nQuantized Model Structure (partial):")
    # print(quantized_model.distilbert.transformer.layer[0].sa_proj)

    # You can now save the quantized model or use it for inference
    # E.g., torch.save(quantized_model.state_dict(), "quantized_distilbert_1bit.pth")

    # Minimal test inference (will likely be very bad accuracy initially)
    print("\nTesting inference with quantized model...")
    inputs = tokenizer("Hello, my name is Rahul.", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = quantized_model(**inputs)
    print("Outputs:", outputs.logits)

    print("\nQuantization and basic inference test complete.")
    print("Next steps: Implement BinaryActivation properly, refine BinaryLinear kernel, benchmark, and evaluate accuracy.")