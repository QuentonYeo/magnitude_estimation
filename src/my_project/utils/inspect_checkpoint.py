#!/usr/bin/env python
"""
Utility script to inspect model checkpoint files and print useful information.

Usage:
    python -m src.my_project.utils.inspect_checkpoint path/to/model_best.pt
    python -m src.my_project.utils.inspect_checkpoint path/to/model_best.pt --verbose
"""

import argparse
import torch
import sys
from pathlib import Path
from typing import Any, Dict


def format_size(num_bytes: int) -> str:
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def count_parameters(state_dict: Dict[str, Any]) -> tuple[int, int]:
    """Count total and trainable parameters from state dict."""
    total_params = 0
    total_elements = 0
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params
            total_elements += 1
    
    return total_params, total_elements


def get_model_architecture_info(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract architecture information from state dict."""
    info = {
        "layers": [],
        "layer_shapes": {},
        "param_groups": {}
    }
    
    # Group layers by prefix
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            # Extract main layer name (e.g., "encoder.0" from "encoder.0.weight")
            parts = name.split('.')
            if len(parts) > 1:
                layer_prefix = '.'.join(parts[:-1])
            else:
                layer_prefix = name
            
            if layer_prefix not in info["param_groups"]:
                info["param_groups"][layer_prefix] = []
            
            info["param_groups"][layer_prefix].append({
                "name": name,
                "shape": list(param.shape),
                "params": param.numel(),
                "dtype": str(param.dtype)
            })
            
            info["layer_shapes"][name] = list(param.shape)
    
    return info


def print_checkpoint_info(checkpoint_path: str, verbose: bool = False):
    """
    Load and print information about a model checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        verbose: If True, print detailed layer-by-layer information
    """
    path = Path(checkpoint_path)
    
    if not path.exists():
        print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("MODEL CHECKPOINT INSPECTOR")
    print("=" * 80)
    print(f"📁 File: {path.name}")
    print(f"📂 Directory: {path.parent}")
    print(f"💾 File size: {format_size(path.stat().st_size)}")
    print()
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception as e:
        try:
            # Try without weights_only for older checkpoints
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            sys.exit(1)
    
    # Determine checkpoint structure
    if isinstance(checkpoint, dict):
        has_wrapper = "model_state_dict" in checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        has_wrapper = False
        state_dict = checkpoint
    
    print("=" * 80)
    print("CHECKPOINT STRUCTURE")
    print("=" * 80)
    
    if has_wrapper:
        print("✓ Wrapped checkpoint (contains metadata)")
        print(f"\n📋 Available keys: {list(checkpoint.keys())}")
        
        # Print metadata
        print("\n" + "-" * 80)
        print("TRAINING METADATA")
        print("-" * 80)
        
        metadata_keys = [
            ("epoch", "Epoch"),
            ("best_epoch", "Best Epoch"),
            ("train_loss", "Training Loss"),
            ("val_loss", "Validation Loss"),
            ("best_val_loss", "Best Validation Loss"),
            ("learning_rate", "Learning Rate"),
            ("optimizer", "Optimizer Type"),
            ("scheduler", "Scheduler Type"),
        ]
        
        for key, label in metadata_keys:
            if key in checkpoint:
                value = checkpoint[key]
                if isinstance(value, float):
                    print(f"{label:25s}: {value:.6f}")
                else:
                    print(f"{label:25s}: {value}")
        
        # Print any additional custom keys
        standard_keys = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", 
                        "epoch", "best_epoch", "train_loss", "val_loss", "best_val_loss",
                        "learning_rate", "optimizer", "scheduler", "train_losses", "val_losses",
                        "train_maes", "val_maes", "learning_rates"}
        
        custom_keys = set(checkpoint.keys()) - standard_keys
        if custom_keys:
            print(f"\n📌 Additional metadata keys: {sorted(custom_keys)}")
    else:
        print("⚠️  Raw state dict (no metadata wrapper)")
    
    # Analyze model architecture
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    
    total_params, total_layers = count_parameters(state_dict)
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Layers: {total_layers}")
    
    # Calculate model size in memory
    model_size_bytes = sum(
        param.numel() * param.element_size() 
        for param in state_dict.values() 
        if isinstance(param, torch.Tensor)
    )
    print(f"Model Size (in memory): {format_size(model_size_bytes)}")
    
    # Get architecture info
    arch_info = get_model_architecture_info(state_dict)
    
    # Print layer groups
    print(f"\n📊 Layer Groups ({len(arch_info['param_groups'])} groups):")
    
    # Group and count parameters by prefix
    prefix_counts = {}
    for prefix, params in arch_info["param_groups"].items():
        total_group_params = sum(p["params"] for p in params)
        prefix_counts[prefix] = {
            "count": len(params),
            "params": total_group_params
        }
    
    # Sort by parameter count
    sorted_prefixes = sorted(prefix_counts.items(), key=lambda x: x[1]["params"], reverse=True)
    
    # Print top 20 layer groups
    print("\nTop 20 layer groups by parameter count:")
    print(f"{'Layer/Module':<50} {'Params':>15} {'# Tensors':>10}")
    print("-" * 80)
    
    for i, (prefix, info) in enumerate(sorted_prefixes[:20]):
        print(f"{prefix:<50} {info['params']:>15,} {info['count']:>10}")
    
    if len(sorted_prefixes) > 20:
        remaining_params = sum(info['params'] for _, info in sorted_prefixes[20:])
        print(f"{'... and ' + str(len(sorted_prefixes) - 20) + ' more':<50} {remaining_params:>15,}")
    
    # Verbose mode: print all layers
    if verbose:
        print("\n" + "=" * 80)
        print("DETAILED LAYER INFORMATION")
        print("=" * 80)
        
        print(f"\n{'Layer Name':<60} {'Shape':<25} {'Params':>15}")
        print("-" * 105)
        
        for name, shape in sorted(arch_info["layer_shapes"].items()):
            params = 1
            for dim in shape:
                params *= dim
            shape_str = str(shape)
            print(f"{name:<60} {shape_str:<25} {params:>15,}")
    
    # Detect model type based on layer names
    print("\n" + "=" * 80)
    print("MODEL TYPE DETECTION")
    print("=" * 80)
    
    layer_names = list(state_dict.keys())
    layer_str = " ".join(layer_names).lower()
    
    detected_types = []
    
    if "umamba" in layer_str or "mamba" in layer_str:
        detected_types.append("🐍 UMamba-based model")
        if "scalar_head" in layer_str and "temporal_head" in layer_str:
            detected_types.append("  → UMamba V3 (triple-head)")
        elif "scalar_head" in layer_str:
            detected_types.append("  → UMamba V2 (scalar output)")
        else:
            detected_types.append("  → UMamba V1 (U-Net style)")
    
    if "lstm" in layer_str:
        detected_types.append("🔄 LSTM-based model")
        if "attention" in layer_str:
            detected_types.append("  → AMAG (LSTM + Attention)")
    
    if "transformer" in layer_str or "multihead" in layer_str:
        detected_types.append("🔀 Transformer-based model")
    
    if "vit" in layer_str or "patch_embed" in layer_str:
        detected_types.append("👁️  Vision Transformer (ViT)")
    
    if "down_branch" in layer_str and "up_branch" in layer_str:
        detected_types.append("🔻 U-Net architecture")
        if "out" in layer_str and any("magnitude" in k for k in layer_names):
            detected_types.append("  → PhaseNet-based magnitude model")
    
    if detected_types:
        print("Detected architecture:")
        for dt in detected_types:
            print(f"  {dt}")
    else:
        print("⚠️  Could not automatically detect model type")
    
    # Print sample layer names to help identify model
    print("\n📝 Sample layer names (first 10):")
    for i, name in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1}. {name}")
    
    if len(state_dict) > 10:
        print(f"  ... and {len(state_dict) - 10} more layers")
    
    print("\n" + "=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch model checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.my_project.utils.inspect_checkpoint model_best.pt
  python -m src.my_project.utils.inspect_checkpoint model_best.pt --verbose
  python -m src.my_project.utils.inspect_checkpoint src/trained_weights/*/model_best.pt
        """
    )
    
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the model checkpoint file (.pt)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed layer-by-layer information"
    )
    
    args = parser.parse_args()
    
    print_checkpoint_info(args.checkpoint_path, args.verbose)


if __name__ == "__main__":
    main()
