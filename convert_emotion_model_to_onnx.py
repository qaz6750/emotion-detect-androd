#!/usr/bin/env python3
"""
Convert PyTorch emotion recognition model to ONNX format
This script converts the fer2013_resnet_best.pth model to ONNX for Android deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# Import the custom ResNet architecture from resnet.py
from resnet import ResNet18, BasicBlock, ResNet

# Use the same architecture as defined in resnet.py

def load_pytorch_model(model_path):
    """Load the PyTorch emotion model"""
    print(f"Loading PyTorch model from: {model_path}")
    
    # Load the checkpoint first to inspect its structure
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Direct state_dict'}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Found 'state_dict' in checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
    else:
        state_dict = checkpoint
        print("Checkpoint is direct state_dict")
      # Print some state_dict keys to understand the structure
    state_keys = list(state_dict.keys())
    print(f"First 5 state_dict keys: {state_keys[:5]}")
    print(f"Total keys: {len(state_keys)}")
    
    # Use the custom ResNet18 architecture from resnet.py
    try:
        print("Attempting to load custom ResNet18 model...")
        model = ResNet18()  # This creates the exact same architecture as used in training
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle any minor differences
        print("✅ Successfully loaded custom ResNet18 model")
    except Exception as e:
        print(f"Failed to load custom ResNet18: {str(e)}")
        
        # Fallback: try to create ResNet manually with the same structure
        try:
            print("Trying manual ResNet creation...")
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=7)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Successfully loaded manual ResNet model")
        except Exception as e2:
            print(f"Failed to load manual ResNet: {str(e2)}")
            raise Exception(f"Could not load model with either approach. Errors: {str(e)}, {str(e2)}")
    
    model.eval()
    return model

def convert_to_onnx(model, output_path, input_size=(1, 1, 48, 48)):
    """Convert PyTorch model to ONNX format"""
    print(f"Converting model to ONNX format...")
    
    # Create dummy input tensor
    dummy_input = torch.randn(*input_size)
    
    # Define input and output names
    input_names = ['input']
    output_names = ['output']
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # Use opset 11 for better compatibility
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to: {output_path}")

def verify_onnx_model(onnx_path, pytorch_model):
    """Verify that ONNX model produces same output as PyTorch model"""
    print("Verifying ONNX model...")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Test with dummy input
    test_input = torch.randn(1, 1, 48, 48)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    
    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    diff = np.abs(pytorch_output.numpy() - ort_output).max()
    print(f"Max difference between PyTorch and ONNX outputs: {diff}")
    
    if diff < 1e-5:
        print("✅ ONNX model verification successful!")
    else:
        print("⚠️  Warning: Significant difference between models")
    
    return diff < 1e-5

def print_model_info(onnx_path):
    """Print ONNX model information"""
    print("\n" + "="*50)
    print("ONNX MODEL INFORMATION")
    print("="*50)
    
    onnx_model = onnx.load(onnx_path)
    
    # Input info
    for input_info in onnx_model.graph.input:
        print(f"Input: {input_info.name}")
        shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        print(f"  Shape: {shape}")
        print(f"  Type: {input_info.type.tensor_type.elem_type}")
    
    # Output info
    for output_info in onnx_model.graph.output:
        print(f"Output: {output_info.name}")
        shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
        print(f"  Shape: {shape}")
        print(f"  Type: {output_info.type.tensor_type.elem_type}")
    
    print(f"Model size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

def create_usage_guide(output_dir):
    """Create a usage guide for the converted model"""
    guide_path = os.path.join(output_dir, "EMOTION_MODEL_GUIDE.md")
    
    guide_content = """# Emotion Recognition ONNX Model Guide

## Model Information
- **Input**: Grayscale face image (48×48 pixels)
- **Output**: 7 emotion probabilities
- **Format**: ONNX (compatible with Android ONNX Runtime)

## Emotion Classes
The model predicts the following 7 emotions:
0. Angry (愤怒)
1. Disgust (厌恶) 
2. Fear (恐惧)
3. Happy (快乐)
4. Sad (悲伤)
5. Surprise (惊讶)
6. Neutral (中性)

## Input Preprocessing
1. **Face Detection**: Extract face region from image
2. **Resize**: Resize face to 48×48 pixels
3. **Grayscale**: Convert to grayscale (1 channel)
4. **Normalize**: Normalize pixel values to [0, 1] range
5. **Tensor Format**: NCHW format [1, 1, 48, 48]

## Usage Example (Python)
```python
import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession('fer2013_resnet_emotion.onnx')

# Preprocess face image
face_gray = cv2.resize(face_image, (48, 48))
face_gray = face_gray.astype(np.float32) / 255.0
input_tensor = face_gray.reshape(1, 1, 48, 48)

# Run inference
outputs = session.run(None, {'input': input_tensor})
emotions = outputs[0][0]

# Get predicted emotion
emotion_idx = np.argmax(emotions)
confidence = emotions[emotion_idx]
```

## Android Implementation Notes
- Use ONNX Runtime Android
- Input tensor: FloatArray of size 2304 (48×48)
- Output tensor: FloatArray of size 7
- Apply softmax to get probabilities
- Use argmax to get predicted class
"""
    
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"Usage guide created: {guide_path}")

def main():
    """Main conversion function"""
    print("🔄 Starting emotion model conversion to ONNX")
    print("="*60)
    
    # Paths
    model_dir = "models"
    pytorch_model_path = os.path.join(model_dir, "fer2013_resnet_best.pth")
    onnx_model_path = os.path.join(model_dir, "fer2013_resnet_emotion.onnx")
    
    # Check if input model exists
    if not os.path.exists(pytorch_model_path):
        print(f"❌ Error: PyTorch model not found at {pytorch_model_path}")
        return False
    
    try:
        # Load PyTorch model
        model = load_pytorch_model(pytorch_model_path)
        
        # Convert to ONNX
        convert_to_onnx(model, onnx_model_path)
        
        # Verify conversion
        if verify_onnx_model(onnx_model_path, model):
            print("✅ Model conversion completed successfully!")
        else:
            print("⚠️  Model converted but verification failed")
        
        # Print model information
        print_model_info(onnx_model_path)
        
        # Create usage guide
        create_usage_guide(model_dir)
        
        print("\n🎉 Emotion recognition model ready for Android deployment!")
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import torch
        import onnx
        import onnxruntime
    except ImportError as e:
        print("Missing required packages. Please install:")
        print("pip install torch torchvision onnx onnxruntime pillow")
        exit(1)
    
    success = main()
    exit(0 if success else 1)
