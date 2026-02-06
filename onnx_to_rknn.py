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


def fix_dfl_transpose(model) -> bool:
    """
    Replace DFL 2-Transpose pattern (opset <=12) with 1-Transpose pattern (opset 19 style).

    RV1106 NPU doesn't support Transpose with perm=[0,3,1,2] or [0,3,2,1], but
    supports perm=[0,2,1,3] (simple dims swap). The opset 12 DFL pattern is:

        Reshape [B,4,16,N] -> Transpose[0,3,1,2] -> [B,N,4,16]
        -> Softmax(axis=3) -> Transpose[0,3,2,1] -> [B,16,4,N]
        -> Conv(16->1) -> [B,1,4,N] -> Reshape [B,4,N]

    I've decided to replace it with the pattern used by opset 19 exports:

        Reshape [B,4,16,N] -> Transpose[0,2,1,3] -> [B,16,4,N]
        -> Softmax(axis=1) -> Conv(16->1) -> [B,1,4,N] -> Reshape [B,4,N]

    This keeps the Conv (NPU-native) and uses only the supported transpose permutation.

    Returns True if the graph was modified.
    """
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np

    graph = model.graph

    # Build lookup maps
    nodes_by_output = {}
    for node in graph.node:
        for out in node.output:
            nodes_by_output[out] = node

    nodes_by_input = {}
    for node in graph.node:
        for inp in node.input:
            if inp not in nodes_by_input:
                nodes_by_input[inp] = []
            nodes_by_input[inp].append(node)

    def get_perm(node):
        for a in node.attribute:
            if a.name == 'perm':
                return list(a.ints)
        return None

    # find DFL 2-transpose pattern:
    # Reshape -> Transpose(perm=[0,3,1,2]) -> Softmax -> Transpose(perm=[0,3,2,1]) -> Conv -> Reshape
    dfl_transpose1 = None
    dfl_transpose2 = None
    dfl_softmax = None
    dfl_conv = None
    dfl_reshape_after = None

    for node in graph.node:
        if node.op_type != 'Transpose':
            continue
        perm = get_perm(node)
        if perm != [0, 3, 1, 2]:
            continue
        if 'dfl' not in node.name.lower():
            continue

        # Check chain: -> Softmax -> Transpose[0,3,2,1] -> Conv
        consumers = nodes_by_input.get(node.output[0], [])
        softmax = next((n for n in consumers if n.op_type == 'Softmax'), None)
        if softmax is None:
            continue

        consumers2 = nodes_by_input.get(softmax.output[0], [])
        transpose2 = next((n for n in consumers2 if n.op_type == 'Transpose'), None)
        if transpose2 is None or get_perm(transpose2) != [0, 3, 2, 1]:
            continue

        consumers3 = nodes_by_input.get(transpose2.output[0], [])
        conv = next((n for n in consumers3 if n.op_type == 'Conv'), None)
        if conv is None:
            continue

        consumers4 = nodes_by_input.get(conv.output[0], [])
        reshape_after = next((n for n in consumers4 if n.op_type == 'Reshape'), None)

        # Find Reshape before first Transpose
        reshape_before = nodes_by_output.get(node.input[0])
        if reshape_before is None or reshape_before.op_type != 'Reshape':
            continue

        dfl_transpose1 = node
        dfl_softmax = softmax
        dfl_transpose2 = transpose2
        dfl_conv = conv
        dfl_reshape_after = reshape_after
        break

    if dfl_transpose1 is None:
        return False

    print(f"  Found DFL 2-Transpose pattern (opset <=12 style)")
    print(f"  Replacing with 1-Transpose pattern (opset 19 style, perm=[0,2,1,3])")

    # Original chain:
    #   reshape_out [B,4,16,N]
    #   -> Transpose1[0,3,1,2] -> tr1_out [B,N,4,16]
    #   -> Softmax(axis=3) -> sm_out [B,N,4,16]
    #   -> Transpose2[0,3,2,1] -> tr2_out [B,16,4,N]
    #   -> Conv(16->1) -> conv_out [B,1,4,N]
    #   -> Reshape -> reshape_out [B,4,N]
    #
    # New chain (reuse existing Conv and Reshape_after, just fix Transpose + Softmax):
    #   reshape_out [B,4,16,N]
    #   -> Transpose[0,2,1,3] -> new_tr_out [B,16,4,N] (swap dims 1,2)
    #   -> Softmax(axis=1) -> new_sm_out [B,16,4,N] (softmax over 16 bins, now in dim 1)
    #   -> Conv(16->1) -> conv_out [B,1,4,N] (reuse existing Conv)
    #   -> Reshape -> reshape_out [B,4,N] (reuse existing Reshape)

    dfl_input = dfl_transpose1.input[0] # [B, 4, 16, N]

    # New Transpose: perm = [0,2,1,3] (swap dims 1 and 2)
    # [B,4,16,N] -> [B,16,4,N]
    new_tr_out = f"{dfl_input}_transpose_fixed"
    new_transpose = helper.make_node(
        'Transpose',
        inputs=[dfl_input],
        outputs=[new_tr_out],
        perm=[0, 2, 1, 3],
        name="dfl_transpose_fixed"
    )

    # New Softmax: axis=1 (the 16-bin dimension, now in position 1 after transpose)
    # reuse Transpose2's output name so Conv input stays valid
    new_sm_out = dfl_transpose2.output[0]
    new_softmax = helper.make_node(
        'Softmax',
        inputs=[new_tr_out],
        outputs=[new_sm_out],
        axis=1,
        name="dfl_softmax_fixed"
    )

    # Conv and Reshape_after stay unchanged - they already consume the right tensors

    # Replace old nodes
    nodes_to_remove = {
        dfl_transpose1.name,
        dfl_softmax.name,
        dfl_transpose2.name,
    }

    new_graph_nodes = []
    inserted = False
    for node in graph.node:
        if node.name in nodes_to_remove:
            if not inserted:
                new_graph_nodes.extend([new_transpose, new_softmax])
                inserted = True
            continue
        new_graph_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_graph_nodes)

    print(f"  Removed: Transpose[0,3,1,2] + Softmax(axis=3) + Transpose[0,3,2,1]")
    print(f"  Added: Transpose[0,2,1,3] + Softmax(axis=1)")
    print(f"  Kept: Conv(16->1) + Reshape (unchanged)")
    return True


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
    opset_version = next((op.version for op in model.opset_import if op.domain == "" or op.domain == "ai.onnx"), 11)
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

            if opset_version >= 13:
                # opset 13 and above: split sizes as second input
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
            else:
                # opset for 11-12: split sizes as attribute
                split_node = helper.make_node(
                    'Split',
                    inputs=[output.name],
                    outputs=[bbox_name, class_name],
                    axis=1,
                    split=[4, num_classes],
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

    # simplify computation graph (constant folding, op fusion)
    simplified_path = None
    onnx_to_load = onnx_path

    try:
        import onnxslim
        import onnx as _onnx

        print("Simplifying ONNX graph...")
        model_proto = _onnx.load(str(onnx_path))
        num_nodes_before = len(model_proto.graph.node)
        simplified = onnxslim.slim(model_proto)

        temp_dir = Path(tempfile.gettempdir())
        simplified_path = temp_dir / f"{onnx_path.stem}_simplified.onnx"
        _onnx.save(simplified, str(simplified_path))
        onnx_to_load = simplified_path
        print(f"  onnxslim OK ({num_nodes_before} -> {len(simplified.graph.node)} nodes)")
    except ImportError:
        print("  onnxslim not installed, skipping simplification")
    except Exception as e:
        print(f"  onnxslim failed: {e}, using original ONNX")

    # and also fix DFL Transpose ops that RV1106 doesn't support.
    # Replaces the 2-Transpose DFL pattern (opset <=12) with a single
    # Transpose perm=[0,2,1,3] (opset 19 style) that the NPU can handle.
    dfl_fixed_path = None
    if platform_info["int8_only"]:
        import onnx as _onnx
        print("Checking DFL head for unsupported Transpose ops...")
        dfl_model = _onnx.load(str(onnx_to_load))
        if fix_dfl_transpose(dfl_model):
            temp_dir = Path(tempfile.gettempdir())
            dfl_fixed_path = temp_dir / f"{onnx_path.stem}_dfl_fixed.onnx"
            _onnx.save(dfl_model, str(dfl_fixed_path))
            onnx_to_load = dfl_fixed_path
            print(f"  Saved DFL-fixed ONNX to: {dfl_fixed_path}")
        else:
            print("  No DFL 2-Transpose pattern found (OK)")

    # Fix YOLOv8 for RV1106: normalize bbox + 3D->4D for zero-copy bug
    fixed_onnx_path = None

    if platform_info["int8_only"]:
        print("Fixing YOLOv8 for RV1106...")
        print("  - Normalizing bbox coordinates (0-1) for better INT8 quantization")
        print("  - Converting 3D->4D output for zero-copy bug workaround")
        fixed_onnx_path = fix_yolov8_for_rv1106(onnx_to_load, width, height)
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

    # Cleanup temp files
    if fixed_onnx_path and fixed_onnx_path.exists():
        fixed_onnx_path.unlink()
    if dfl_fixed_path and dfl_fixed_path.exists():
        dfl_fixed_path.unlink()
    if simplified_path and simplified_path.exists():
        simplified_path.unlink()

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
