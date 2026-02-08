import onnx
model = onnx.load('yolov8n.onnx')
for n in model.graph.node:
    if n.op_type == 'Sigmoid':
        print(f'{n.name}: inputs={list(n.input)} outputs={list(n.output)}')
