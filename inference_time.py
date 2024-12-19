import time
from models import AlexNet,Resnet_18,EfficientNet_B0,EfficientNet_V2_S,EfficientNet_V2_L,VisionTransformer_B_16,VisionTransformer_L_16,VisionTransformer_H_14
import torch

def measure_inference_time(model, input_tensor, num_warmup=10, num_runs=100):
    model = model.eval().to(input_tensor.device)
    
    # Warmup runs
    print("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Measure inference time
    total_time = 0
    print("Measuring inference time...")
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            total_time += time.perf_counter() - start_time
    
    return (total_time / num_runs) * 1000  # Convert to milliseconds

def main():
    torch.set_float32_matmul_precision('medium')
    device = 'cuda:0'
    
    # List of models to test
    models = [
        
        ('AlexNet', AlexNet()),
        ('ResNet-18', Resnet_18()),
        ('EfficientNet-B0', EfficientNet_B0()),
        ('EfficientNetV2-S', EfficientNet_V2_S()),
        ('EfficientNetV2-L', EfficientNet_V2_L()),
        ('VisionTransformer-B_16', VisionTransformer_B_16()),
        ('VisionTransformer-L_16', VisionTransformer_L_16()),
        ('VisionTransformer-H_14', VisionTransformer_H_14())
    ]
    
    # Create random input tensor
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Test each model
    for model_name, model in models:
        print(f"\nTesting {model_name}...")
        avg_time = measure_inference_time(model, input_tensor,num_warmup=100, num_runs=1000)
        print(f"{model_name} average inference time: {avg_time:.2f} ms")

if __name__ == '__main__':
    main()