"""
Export NeMo ASR model to ONNX format with optimizations
This script demonstrates the model optimization process
"""
import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import logging
from onnx import shape_inference
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NemoToONNXExporter:
    """Export and optimize NeMo ASR model to ONNX format"""
    
    def __init__(self, nemo_model_path: str = None):
        """
        Initialize exporter
        
        Note: Since we're using a pre-exported ONNX model from NGC,
        this demonstrates the process that would be used with a .nemo file
        """
        self.nemo_model_path = nemo_model_path
        
    def export_to_onnx(self, output_path: str = "asr_model.onnx"):
        """
        Export NeMo model to ONNX format
        
        In practice, this would:
        1. Load the .nemo checkpoint
        2. Extract the encoder and decoder
        3. Trace the model with example inputs
        4. Export to ONNX
        """
        logger.info("=== NeMo to ONNX Export Process ===")
        logger.info("This demonstrates how to export a NeMo model to ONNX format")
        
        # Demonstration of what the export process would look like:
        logger.info("\nStep 1: Load NeMo model")
        logger.info("```python")
        logger.info("import nemo.collections.asr as nemo_asr")
        logger.info("model = nemo_asr.models.EncDecCTCModelBPE.restore_from('model.nemo')")
        logger.info("model.eval()")
        logger.info("```")
        
        logger.info("\nStep 2: Prepare dummy inputs")
        logger.info("```python")
        logger.info("batch_size = 1")
        logger.info("time_steps = 1000  # ~10 seconds at 16kHz")
        logger.info("n_mels = 80")
        logger.info("dummy_input = torch.randn(batch_size, n_mels, time_steps)")
        logger.info("dummy_length = torch.tensor([time_steps])")
        logger.info("```")
        
        logger.info("\nStep 3: Export to ONNX")
        logger.info("```python")
        logger.info("torch.onnx.export(")
        logger.info("    model,")
        logger.info("    (dummy_input, dummy_length),")
        logger.info("    'model.onnx',")
        logger.info("    input_names=['audio_signal', 'length'],")
        logger.info("    output_names=['logprobs'],")
        logger.info("    dynamic_axes={")
        logger.info("        'audio_signal': {0: 'batch', 2: 'time'},")
        logger.info("        'length': {0: 'batch'},")
        logger.info("        'logprobs': {0: 'batch', 1: 'time'}")
        logger.info("    },")
        logger.info("    opset_version=14,")
        logger.info("    do_constant_folding=True")
        logger.info(")")
        logger.info("```")
        
        logger.info(f"\nNote: Using pre-exported ONNX model from NVIDIA NGC")
        return output_path
    
    def optimize_onnx_with_ort(self, input_path: str, output_path: str = "asr_model_optimized.onnx"):
        """Apply ONNX Runtime optimizations for inference"""
        logger.info(f"\n=== Optimizing ONNX model: {input_path} ===")
        
        # Use ONNX Runtime's optimization
        import onnxruntime as ort
        
        # Set optimization options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = output_path
        
        # Load and optimize
        logger.info("Applying ONNX Runtime optimizations...")
        session = ort.InferenceSession(input_path, sess_options)
        
        logger.info(f"✓ Optimized model saved to {output_path}")
        
        # Verify the optimized model
        self.verify_onnx_model(output_path)
        
        return output_path
    
    def analyze_model(self, model_path: str):
        """Analyze ONNX model structure and properties"""
        logger.info(f"\n=== Analyzing model: {model_path} ===")
        
        # Load model
        model = onnx.load(model_path)
        
        # Basic info
        logger.info(f"Model producer: {model.producer_name}")
        logger.info(f"Model version: {model.producer_version}")
        logger.info(f"ONNX version: {model.opset_import[0].version}")
        
        # Count operations
        op_types = {}
        for node in model.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        
        logger.info("\nOperation counts:")
        for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {op_type}: {count}")
        
        # Model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"\nModel size: {model_size:.2f} MB")
        
        # Input/Output info
        logger.info("\nInputs:")
        for input in model.graph.input:
            logger.info(f"  {input.name}: {[d.dim_value if d.HasField('dim_value') else d.dim_param for d in input.type.tensor_type.shape.dim]}")
        
        logger.info("\nOutputs:")
        for output in model.graph.output:
            logger.info(f"  {output.name}: {[d.dim_value if d.HasField('dim_value') else d.dim_param for d in output.type.tensor_type.shape.dim]}")
    
    def verify_onnx_model(self, model_path: str):
        """Verify ONNX model is valid and runnable"""
        logger.info(f"\n=== Verifying ONNX model ===")
        
        # Check model validity
        model = onnx.load(model_path)
        try:
            onnx.checker.check_model(model)
            logger.info("✓ Model structure is valid")
        except Exception as e:
            logger.error(f"✗ Model validation failed: {e}")
            return False
        
        # Test inference
        session = ort.InferenceSession(model_path)
        
        # Get input details
        inputs = session.get_inputs()
        logger.info(f"Model inputs: {[(i.name, i.shape) for i in inputs]}")
        
        # Create dummy input
        dummy_audio = np.random.randn(1, 80, 100).astype(np.float32)
        dummy_length = np.array([100], dtype=np.int64)
        
        # Run inference
        try:
            outputs = session.run(None, {
                'audio_signal': dummy_audio,
                'length': dummy_length
            })
            logger.info(f"✓ Inference successful. Output shape: {outputs[0].shape}")
            return True
        except Exception as e:
            logger.error(f"✗ Inference failed: {e}")
            return False
    
    def benchmark_model(self, model_path: str, num_runs: int = 100):
        """Benchmark model performance"""
        import time
        
        logger.info(f"\n=== Benchmarking model performance ===")
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return
        
        session = ort.InferenceSession(model_path)
        
        # Prepare inputs for different durations
        test_cases = [
            ("1 second", np.random.randn(1, 80, 100).astype(np.float32)),
            ("5 seconds", np.random.randn(1, 80, 500).astype(np.float32)),
            ("10 seconds", np.random.randn(1, 80, 1000).astype(np.float32))
        ]
        
        for name, dummy_audio in test_cases:
            dummy_length = np.array([dummy_audio.shape[2]], dtype=np.int64)
            
            # Warmup
            for _ in range(10):
                session.run(None, {'audio_signal': dummy_audio, 'length': dummy_length})
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.time()
                session.run(None, {'audio_signal': dummy_audio, 'length': dummy_length})
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            
            logger.info(f"\n{name} audio:")
            logger.info(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
            logger.info(f"  Throughput: {1000/avg_time:.2f} inferences/second")
            logger.info(f"  Real-time factor: {float(name.split()[0])*1000/avg_time:.2f}x")

    def export_optimization_guide(self):
        """Export a guide for model optimization"""
        guide = """
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
"""
        
        with open("OPTIMIZATION_GUIDE.md", "w") as f:
            f.write(guide)
        
        logger.info("\n✓ Optimization guide exported to OPTIMIZATION_GUIDE.md")

def main():
    """Main export and optimization pipeline"""
    exporter = NemoToONNXExporter()
    
    # Show export process
    exporter.export_to_onnx()
    
    # If model exists, analyze and optimize it
    model_path = "model/asr_model_hi.onnx"
    if os.path.exists(model_path):
        # Analyze original model
        exporter.analyze_model(model_path)
        
        # Optimize model
        optimized_path = "model/asr_model_hi_optimized.onnx"
        exporter.optimize_onnx_with_ort(model_path, optimized_path)
        
        # Benchmark both models
        logger.info("\n=== Comparing Performance ===")
        logger.info("\nOriginal model:")
        exporter.benchmark_model(model_path, num_runs=50)
        
        if os.path.exists(optimized_path):
            logger.info("\nOptimized model:")
            exporter.benchmark_model(optimized_path, num_runs=50)
    else:
        logger.info(f"\nModel not found at {model_path}")
        logger.info("Please place your ONNX model in the model/ directory")
    
    # Export optimization guide
    exporter.export_optimization_guide()

if __name__ == "__main__":
    main()
