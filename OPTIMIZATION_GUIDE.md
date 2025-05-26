
# ONNX Model Optimization Guide

## 1. Export Optimizations (during torch.onnx.export)
- Use `do_constant_folding=True` to fold constant operations
- Set appropriate `opset_version` (14 or higher recommended)
- Use `dynamic_axes` for variable-length inputs

## 2. Graph Optimizations
- Fuse operations (Conv+BN, Linear+Activation)
- Eliminate redundant operations
- Constant folding and propagation
- Dead code elimination

## 3. Quantization (for further optimization)
```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input='model.onnx',
    model_output='model_quantized.onnx',
    weight_type=QuantType.QUInt8
)
```

## 4. Runtime Optimizations
- Use appropriate execution providers (CPU, CUDA, TensorRT)
- Enable graph optimizations in ONNX Runtime
- Use IOBinding for zero-copy inference
- Batch multiple requests when possible

## 5. Model-Specific Optimizations for ASR
- Use streaming inference for long audio
- Implement VAD (Voice Activity Detection) to skip silent parts
- Cache mel-spectrogram computation
- Use lower precision (FP16) where possible
