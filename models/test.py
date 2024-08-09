import onnx
model = onnx.load('simplified_24.onnx')

node = model.graph.node
node[4].op_type = 'Add'

onnx.checker.check_model(model)
