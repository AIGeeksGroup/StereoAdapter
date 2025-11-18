import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os

# Add the project path to sys.path
sys.path.append('/home/ywan0794/TiO-Depth_pytorch')

from models.decoders.gru_decoder import GRUDecoder, BasicMultiUpdateBlock

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_layer_memory(layer):
    """Estimate memory usage of a layer based on its parameters"""
    total_params = 0
    for param in layer.parameters():
        total_params += param.numel()
    # Each parameter is typically float32 (4 bytes)
    memory_mb = (total_params * 4) / (1024 * 1024)
    return total_params, memory_mb

def inspect_gru_decoder():
    print("=" * 80)
    print("GRU Decoder Analysis")
    print("=" * 80)
    
    # Initialize GRUDecoder with the same parameters as in your code
    decoder = GRUDecoder(
        corr_implementation="reg",
        corr_radius=4,
        corr_levels=4,
        n_gru_layers=4,
        context_dims=[128, 128, 128, 128]
    )
    
    print(f"\nTotal parameters: {count_parameters(decoder):,}")
    
    # Inspect the update_block specifically
    update_block = decoder.update_block
    print(f"\nUpdate block parameters: {count_parameters(update_block):,}")
    
    # Inspect the mask layer specifically
    mask_layer = update_block.mask
    mask_params, mask_memory = analyze_layer_memory(mask_layer)
    print(f"\nMask layer:")
    print(f"  Parameters: {mask_params:,}")
    print(f"  Memory (MB): {mask_memory:.2f}")
    
    # Show the architecture of the mask layer
    print(f"\nMask layer architecture:")
    for i, layer in enumerate(mask_layer):
        if isinstance(layer, nn.Conv2d):
            print(f"  Layer {i}: Conv2d(in_channels={layer.in_channels}, "
                  f"out_channels={layer.out_channels}, kernel_size={layer.kernel_size})")
    
    # Calculate the problematic factor
    n_downsample = update_block.args.n_downsample
    factor = 2 ** n_downsample
    mask_output_channels = (factor ** 2) * 9
    
    print(f"\nProblem Analysis:")
    print(f"  n_downsample: {n_downsample}")
    print(f"  factor: 2^{n_downsample} = {factor}")
    print(f"  factor^2: {factor}^2 = {factor**2}")
    print(f"  Mask output channels: {factor**2} * 9 = {mask_output_channels:,}")
    
    # Test with a smaller n_downsample value
    print("\n" + "=" * 80)
    print("Testing with corrected n_downsample values")
    print("=" * 80)
    
    for n_down in [1, 2, 3, 4]:
        raft_args = SimpleNamespace(
            corr_radius=4,
            corr_levels=4,
            n_gru_layers=4,
            encoder_output_dim=128,
            n_downsample=n_down
        )
        
        test_block = BasicMultiUpdateBlock(raft_args, hidden_dims=[128, 128, 128, 128])
        mask_params, mask_memory = analyze_layer_memory(test_block.mask)
        
        factor = 2 ** n_down
        output_channels = (factor ** 2) * 9
        
        print(f"\nn_downsample={n_down}:")
        print(f"  factor={factor}, output_channels={output_channels}")
        print(f"  Mask parameters: {mask_params:,}")
        print(f"  Mask memory: {mask_memory:.2f} MB")
    
    # Estimate memory for forward pass
    print("\n" + "=" * 80)
    print("Memory usage estimation for forward pass")
    print("=" * 80)
    
    batch_size = 1
    height, width = 480, 640  # Example resolution
    
    for n_down in [1, 2, 3, 4, 7]:
        factor = 2 ** n_down
        output_channels = (factor ** 2) * 9
        
        # Memory for output tensor (assuming float32)
        output_memory_gb = (batch_size * output_channels * height * width * 4) / (1024**3)
        
        print(f"\nn_downsample={n_down} (output_channels={output_channels:,}):")
        print(f"  Output tensor shape: [{batch_size}, {output_channels}, {height}, {width}]")
        print(f"  Output tensor memory: {output_memory_gb:.2f} GB")
        
        if output_memory_gb > 20:
            print(f"  ⚠️  WARNING: This will likely cause OOM on a 24GB GPU!")

if __name__ == "__main__":
    inspect_gru_decoder()