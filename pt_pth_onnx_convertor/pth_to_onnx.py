import torch
import torchvision.models as models

# Convert mobilenet_v3_large.pth to ONNX
def convert_mobilenet_v3_to_onnx():
    # Load MobileNetV3 Large architecture (updated to use weights=None)
    model = models.mobilenet_v3_large(weights=None)

    # Adjust the classifier for your number of classes (e.g., 5 classes for banana ripeness: 5D to 1D)
    num_classes = 5  # Adjust this based on your training
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

    # Load your custom weights from mobilenet_v3_large.pth, mapping to CPU
    state_dict = torch.load('mobilenet_v3_large.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set the model to evaluation mode

    # Dummy input (MobileNetV3 expects 224x224 images)
    dummy_input = torch.randn(1, 3, 224, 224)  # pylint: disable=no-member

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "mobilenet_v3_large.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print("MobileNetV3 Large model converted to ONNX: mobilenet_v3_large.onnx")

if __name__ == "__main__":
    convert_mobilenet_v3_to_onnx()