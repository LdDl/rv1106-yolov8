#!/usr/bin/env python3
"""
Convert ONNX model to RKNN format for RV1106 (LuckFox Pico).

Requirements:
    pip install -r requirements.txt

Usage:
    python onnx_to_rknn.py yolov8n.onnx --dataset images/  # with calibration, auto-detect size from ONNX
    python onnx_to_rknn.py yolov8n.onnx --size 416x256     # override size
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Supported target platforms and their capabilities
SUPPORTED_PLATFORMS = {
    "rv1106": {"int8_only": True, "description": "LuckFox Pico Ultra W"},
    # "rv1103": {"int8_only": True, "description": "LuckFox Pico"},
}


def fix_yolov8_for_rv1106(onnx_path: Path, width: int, height: int) -> Path | None:
    """
    Fix YOLOv8 ONNX model for RV1106:
    1. Normalize bbox coordinates (divide by image size) for better INT8 quantization
    2. Convert 3D output [1, 8, N] to 4D [1, 8, N, 1] for zero-copy bug workaround

    The problem: bbox coords range 0-500, class scores range 0-0.25
    With single INT8 scale, class scores get crushed to a single value.
    Solution: Split tensor, normalize only bbox, concat back.

    Returns: Path to modified ONNX file, or None if no changes needed.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np

    model = onnx.load(str(onnx_path))
    graph = model.graph
    modified = False

    new_outputs = []
    nodes_to_add = []
    initializers_to_add = []

    for output in graph.output:
        shape = output.type.tensor_type.shape
        dims = [d.dim_value for d in shape.dim]

        # Check if YOLOv8 output: [1, 8, N] or similar
        if len(dims) == 3 and dims[0] == 1 and dims[1] in (8, 84) and dims[2] > 100:
            num_features = dims[1]
            num_classes = num_features - 4
            num_predictions = dims[2]
            print(f"  Fixing YOLOv8 output '{output.name}': {dims}")
            print(f"    - Splitting into bbox (4) and class ({num_classes}) channels")
            print(f"    - Normalizing bbox by {width}x{height} (range 0-1)")
            print(f"    - Applying Sigmoid to class (range 0-1, same as bbox)")
            print(f"    - Concat back and reshape to 4D for zero-copy bug")

            # Step 1: Split into bbox [1,4,N] and class [1,num_classes,N]
            bbox_name = f"{output.name}_bbox"
            class_name = f"{output.name}_class"

            # Split sizes
            split_sizes_name = f"{output.name}_split_sizes"
            split_sizes = np.array([4, num_classes], dtype=np.int64)
            split_sizes_tensor = numpy_helper.from_array(split_sizes, split_sizes_name)
            initializers_to_add.append(split_sizes_tensor)

            split_node = helper.make_node(
                'Split',
                inputs=[output.name, split_sizes_name],
                outputs=[bbox_name, class_name],
                axis=1,
                name=f"split_{output.name}"
            )
            nodes_to_add.append(split_node)

            # Step 2: Normalize bbox only with Mul (1/width, 1/height)
            # Using Mul instead of Div for better RKNN compatibility
            # Shape must be [1, 4, 1] to broadcast with [1, 4, N]
            bbox_scale = np.array([[[
                1.0 / width, # cx
                1.0 / height, # cy
                1.0 / width, # w
                1.0 / height # h
            ]]], dtype=np.float32).transpose(0, 2, 1)

            bbox_scale_name = f"{output.name}_bbox_scale"
            bbox_scale_tensor = numpy_helper.from_array(bbox_scale, bbox_scale_name)
            initializers_to_add.append(bbox_scale_tensor)

            bbox_norm_name = f"{output.name}_bbox_norm"
            mul_node = helper.make_node(
                'Mul',
                inputs=[bbox_name, bbox_scale_name],
                outputs=[bbox_norm_name],
                name=f"normalize_bbox_{output.name}"
            )
            nodes_to_add.append(mul_node)

            # Step 3: Apply Sigmoid to class scores so it in 0-1 range
            # This makes class range similar to normalized bbox (0-1)
            # So single INT8 scale/zp works for both!
            class_sigmoid_name = f"{output.name}_class_sigmoid"
            sigmoid_node = helper.make_node(
                'Sigmoid',
                inputs=[class_name],
                outputs=[class_sigmoid_name],
                name=f"sigmoid_class_{output.name}"
            )
            nodes_to_add.append(sigmoid_node)

            # Step 4: Concat bbox_norm and class_sigmoid back together
            concat_name = f"{output.name}_concat"
            concat_node = helper.make_node(
                'Concat',
                inputs=[bbox_norm_name, class_sigmoid_name],
                outputs=[concat_name],
                axis=1,
                name=f"concat_{output.name}"
            )
            nodes_to_add.append(concat_node)

            # Step 5: Reshape to 4D for zero-copy bug workaround
            # [1, 8, N, 1]
            new_shape = [dims[0], dims[1], num_predictions, 1]
            shape_name = f"{output.name}_shape_4d"
            shape_tensor = helper.make_tensor(
                shape_name,
                TensorProto.INT64,
                [len(new_shape)],
                new_shape
            )
            initializers_to_add.append(shape_tensor)

            reshaped_name = f"{output.name}_4d"
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[concat_name, shape_name],
                outputs=[reshaped_name],
                name=f"reshape_{output.name}_to_4d"
            )
            nodes_to_add.append(reshape_node)

            # Single output with 4D shape
            new_output = helper.make_tensor_value_info(
                reshaped_name,
                output.type.tensor_type.elem_type,
                new_shape
            )
            new_outputs.append(new_output)
            modified = True
        else:
            new_outputs.append(output)

    if not modified:
        return None

    # Add new nodes and initializers
    graph.node.extend(nodes_to_add)
    graph.initializer.extend(initializers_to_add)

    # Replace outputs
    del graph.output[:]
    graph.output.extend(new_outputs)

    # Save to temp file
    temp_dir = Path(tempfile.gettempdir())
    fixed_path = temp_dir / f"{onnx_path.stem}_normalized_4d.onnx"
    onnx.save(model, str(fixed_path))

    print(f"  Saved fixed ONNX to: {fixed_path}")
    return fixed_path


def get_onnx_input_size(onnx_path: Path) -> tuple[int, int]:
    """Get input size (width, height) from ONNX model."""
    import onnx
    model = onnx.load(str(onnx_path))
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    # Shape is [batch, channels, height, width]
    height = input_shape[2].dim_value
    width = input_shape[3].dim_value
    return width, height


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string like '640' or '416x256' into (width, height)."""
    if 'x' in size_str:
        parts = size_str.lower().split('x')
        return int(parts[0]), int(parts[1])
    else:
        size = int(size_str)
        return size, size


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN for RV1106")
    parser.add_argument("onnx_model", help="Path to ONNX model")
    parser.add_argument("--output", "-o", help="Output RKNN path (default: same name as input)")
    parser.add_argument("--size", default=None, help="Input size: WxH (e.g., 416x256) or single int. Auto-detected from ONNX if not specified.")
    parser.add_argument("--dataset", help="Calibration dataset directory for quantization")
    parser.add_argument("--max-images", type=int, default=0, help="Max calibration images (0 = all)")
    parser.add_argument("--platform", default="rv1106", help="Target platform (default: rv1106)")
    args = parser.parse_args()

    if args.platform not in SUPPORTED_PLATFORMS:
        supported = ", ".join(SUPPORTED_PLATFORMS.keys())
        print(f"Error: unsupported platform '{args.platform}'")
        print(f"Supported: {supported}")
        return 1

    onnx_path = Path(args.onnx_model)
    if not onnx_path.exists():
        print(f"Error: ONNX file not found: {onnx_path}")
        return 1

    # Get size: from argument or auto-detect from ONNX
    if args.size:
        width, height = parse_size(args.size)
    else:
        try:
            width, height = get_onnx_input_size(onnx_path)
            print(f"Auto-detected input size from ONNX: {width}x{height}")
        except Exception as e:
            print(f"Error: Could not auto-detect size from ONNX: {e}")
            print("Please specify --size manually")
            return 1

    platform_info = SUPPORTED_PLATFORMS[args.platform]
    requires_quantization = platform_info["int8_only"]

    try:
        from rknn.api import RKNN
    except ImportError:
        print("Error: rknn-toolkit2 not installed")
        print()
        print("Install with:")
        print("  pip install rknn-toolkit2")
        print()
        print("Or from Rockchip repo:")
        print("  https://github.com/airockchip/rknn-toolkit2")
        print()
        print("Note: rknn-toolkit2 only works on x86_64 Linux!")
        return 1

    if args.output:
        rknn_path = Path(args.output)
    else:
        rknn_path = onnx_path.with_suffix(".rknn")

    print(f"Converting: {onnx_path}")
    print(f"Target platform: {args.platform}")
    print(f"Input size: {width}x{height}")
    print(f"Quantization: INT8" + (" (required for this platform)" if requires_quantization else ""))
    print()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Config
    print("Configuring RKNN...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
    )

    # Fix YOLOv8 for RV1106: normalize bbox + 3D->4D for zero-copy bug
    onnx_to_load = onnx_path
    fixed_onnx_path = None

    if platform_info["int8_only"]:
        print("Fixing YOLOv8 for RV1106...")
        print("  - Normalizing bbox coordinates (0-1) for better INT8 quantization")
        print("  - Converting 3D->4D output for zero-copy bug workaround")
        fixed_onnx_path = fix_yolov8_for_rv1106(onnx_path, width, height)
        if fixed_onnx_path:
            onnx_to_load = fixed_onnx_path
            print("  Using fixed ONNX")
        else:
            print("  No changes needed")

    # Load ONNX model
    print("Loading ONNX model...")
    ret = rknn.load_onnx(model=str(onnx_to_load))
    if ret != 0:
        print(f"Failed to load ONNX model: {ret}")
        return 1

    # Build
    print("Building RKNN model...")

    # Always INT8 for rv1106/rv1103
    do_quantization = requires_quantization
    dataset_file = (onnx_path.parent / "dataset.txt").resolve()
    temp_images = []

    if args.dataset:
        # Prepare calibration dataset
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset directory not found: {dataset_path}")
            return 1

        # Create dataset file
        images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
        if not images:
            print(f"Error: No images found in {dataset_path}")
            return 1

        # Use all images or limit if specified
        if args.max_images > 0:
            images = images[:args.max_images]

        with open(dataset_file, "w") as f:
            for img in images:
                f.write(f"{img.resolve()}\n")

        print(f"Using {len(images)} images for calibration")
    elif do_quantization:
        # Generate random calibration images (not ideal but works)
        import numpy as np
        from PIL import Image

        print("No dataset provided, generating random calibration images...")
        print("(For better accuracy, provide real images with --dataset)")

        for i in range(3):
            # PIL uses (width, height), numpy uses (height, width, channels)
            img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = onnx_path.parent / f"_calib_{i}.png"
            img.save(img_path)
            temp_images.append(img_path)

        with open(dataset_file, "w") as f:
            for img_path in temp_images:
                f.write(f"{img_path.resolve()}\n")

    if do_quantization:
        ret = rknn.build(
            do_quantization=True,
            dataset=str(dataset_file),
        )
    else:
        ret = rknn.build(do_quantization=False)

    # Cleanup
    if dataset_file.exists():
        dataset_file.unlink()
    for img_path in temp_images:
        if img_path.exists():
            img_path.unlink()

    if ret != 0:
        print(f"Failed to build RKNN model: {ret}")
        return 1

    # Export
    print(f"Exporting to: {rknn_path}")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        print(f"Failed to export RKNN model: {ret}")
        return 1

    print("Cleanup")

    rknn.release()

    # Cleanup temp file
    if fixed_onnx_path and fixed_onnx_path.exists():
        fixed_onnx_path.unlink()

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
