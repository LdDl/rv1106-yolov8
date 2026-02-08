# rv1106-yolov8

Convert and deploy YOLOv8 models on RV1106 NPU (LuckFox Pico Ultra W) with INT8 quantization.

Handles RV1106-specific issues automatically:
- **DFL head fix** - replaces unsupported 2-Transpose DFL pattern (opset <=12) with 1-Transpose (opset 19 style) that the NPU can handle
- **Graph simplification** - constant folding and op fusion via `onnxslim` (same as Ultralytics)
- **Zero-copy bug** - 3D output tensors only write first half of data; workaround reshapes to 4D
- **INT8 quantization** - bbox coords (0-500) and class scores (0-1) have vastly different ranges; solved by normalizing bbox to 0-1 and applying Sigmoid to class scores inside the ONNX graph
- **Sigmoid on class scores** - if sigmoid has already represented in the ONNX, the converter detects and skips adding another one, preventing double-sigmoid issues
Target platform: **RV1106** (LuckFox Pico Ultra W)

## Requirements

- Python 3.10-3.12 (rknn-toolkit2 is not yet available on PyPI for 3.13+)
- ONNX model exported from Ultralytics, or any custom trained.

## Setup

### Python version (if you have Python 3.13+)

rknn-toolkit2 requires Python 3.10-3.12. Use pyenv:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install 3.12.8
pyenv shell 3.12.8
```

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Convert ONNX to RKNN

```bash
source .venv/bin/activate

# Auto-detect input size from ONNX
python onnx_to_rknn.py model.onnx --dataset path/to/calibration/images/

# Override input size
python onnx_to_rknn.py model.onnx --size 416x256 --dataset path/to/images/ --max-images 100
```

Output: `model.rknn` (same directory as input ONNX)

#### onnx_to_rknn.py flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `onnx_model` | | *(required)* | Path to ONNX model |
| `--output` | `-o` | same as input | Output RKNN path |
| `--size` | | auto-detect | Input size `WxH` (e.g. `416x256`). Auto-detected from ONNX if not specified |
| `--dataset` | | | Calibration images directory for INT8 quantization |
| `--max-images` | | `0` (all) | Max number of calibration images to use |
| `--platform` | | `rv1106` | Target platform (`rv1106` is only option currently) |

### Calibration dataset

INT8 quantization needs a representative set of images to determine the optimal value ranges. The script expects a flat directory with `.jpg` and/or `.png` files:

```
calibration_images/
|-- img001.jpg
|-- img002.jpg
|-- img003.png
`-- ...
```

No annotations or labels needed - only the images themselves.
Calibration is working in following way:
- the converter runs the model in FP32 on these images, records the range of values at each layer, and picks the best `scale` and `zero_point` to map that range into 256 INT8 values with minimal loss.
- labels (a.k.a annotations) aren't involved because we only care about what activation values actually occur, not whether the prediction is correct.

The images should be **similar to what the model will see in production** (same domain, lighting, camera angle). If the calibration images have a different distribution (e.g. random noise), the quantization ranges will be wrong and accuracy will suffer.

50-200 images is usually enough. If `--dataset` is not provided, the script falls back to random noise images, which works but gives worse accuracy.

### Deploy to device

In my case: LuckFox Pico Ultra W
```bash
# Community edition Ubuntu
scp model.rknn pico@172.32.0.70:~/
# or for buildroot:
scp model.rknn pico@172.32.0.93:~/
```

## How it works

`onnx_to_rknn.py` processes the ONNX graph before RKNN conversion:

1. **Simplify** the ONNX graph (constant folding, op fusion)
2. **Fix DFL head** - replace 2-Transpose pattern (opset <=12) with 1-Transpose (opset 19 style). Without this, RV1106 fails with `unsupport cpu Transpose op`
3. **Split** output `[1, C, N]` into bbox `[1, 4, N]` and class `[1, num_classes, N]`
4. **Normalize** bbox by dividing by image dimensions (values become 0-1)
5. **Sigmoid** on class scores (values become 0-1)
6. **Concat** back to `[1, C, N]`
7. **Reshape** to `[1, C, N, 1]` (4D) to avoid zero-copy bug

On-device inference must account for:
- **NC1HWC2 layout** - RKNN packs channels into blocks of 16: `offset = prediction * C2 + feature`
- **INT8 dequantization** - `value = (raw_int8 - zp) * scale`
- **Confidence threshold > 0.5** - since `sigmoid(0) = 0.5`, anything below is noise

## Troubleshooting

### "No module named 'rknn'"
Activate the venv: `source .venv/bin/activate`

### Python version error
rknn-toolkit2 requires Python 3.10-3.12. Use pyenv.

### "unsupport cpu Transpose op" on device
Two possible causes:

1. **DFL head** (opset <=12 models): The script auto-fixes this by replacing the 2-Transpose DFL pattern with a 1-Transpose version the NPU supports. If exporting from Ultralytics, use `opset=19` to avoid this entirely.

2. **Model too large for RV1106**: The RV1106 NPU has limited memory. Standard YOLOv8 at 640x640 with 80 classes (8400 predictions, 84 channels) is too large - the RKNN runtime adds internal Transpose ops during optimization that fail at runtime. So **use `imgsz=320` or smaller** (gives ~2100 predictions, which works I believe). Custom models with fewer classes may work at larger sizes (e.g. 4-class model works at 416x256).

### Graph simplification

Done via `onnxslim`.

### No detections / low accuracy
Use more representative calibration images with `--dataset` and `--max-images`.

## Quick start (YOLOv8n trained on COCO)

```bash
source .venv/bin/activate

# 1. Download pretrained YOLOv8n weights and export to ONNX (opset=19, 320x320)
./download_v8n.sh

# 2. Download COCO128 calibration images (128 images, ~7 MB)
./download_dataset.sh

# 3. Convert to RKNN with calibration
python onnx_to_rknn.py yolov8n.onnx --dataset calibration_images/
```

After you get *.rknn weights which you can deploy to RV1106 device and use with an inference tool you prefer.

## Future work

### Darknet (YOLOv3-tiny / YOLOv4-tiny) support

rknn-toolkit2 natively supports Darknet format via `load_darknet()`:

```python
rknn = RKNN()
rknn.config(target_platform='rv1106')
rknn.load_darknet(model='yolov4-tiny.cfg', weight='yolov4-tiny.weights')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('yolov4-tiny.rknn')
```

Tiny variants have simpler architectures (fewer layers, fewer parameters) and should run significantly faster than YOLOv8n on the RV1106 NPU. This is worth exploring when detection speed matters more than accuracy.

Note: the Darknet-to-ONNX-to-RKNN path is also viable and gives more control over the graph (e.g. removing post-processing, reshaping outputs for the zero-copy bug workaround).

## Links

- [rknn_model_zoo/yolov8](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8) - reference YOLOv8 implementation for RKNN
- [Ultralytics RKNN Integration](https://docs.ultralytics.com/integrations/rockchip-rknn/)
- [airockchip/rknn-toolkit2#444](https://github.com/airockchip/rknn-toolkit2/issues/444) - simulator output correct, device output wrong
- [rockchip-linux/rknn-toolkit2#333](https://github.com/rockchip-linux/rknn-toolkit2/issues/333) - different output on rv1106 and PC
- [ultralytics/ultralytics#4097](https://github.com/ultralytics/ultralytics/issues/4097) - how to effectively quantize YOLOv8 to INT8
- [Deploying YOLOv8 on RK3566](https://dev.to/zediot/deploying-yolov8-on-rk3566-using-rknn-toolkit-notes-pitfalls-and-benchmarks-12fj) - practical notes on RKNN INT8 pitfalls
