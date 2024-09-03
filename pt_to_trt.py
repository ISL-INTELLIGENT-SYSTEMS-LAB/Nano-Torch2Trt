import sys
import torch
import torchvision
import torch2trt
import pathlib
import argparse

def get_resnet_model(checkpoint, layers):
    
    if layers == 18:
        return torchvision.models.quantization.resnet18(pretrained=False, quantize=False)
    elif layers == 50:
        return torchvision.models.quantization.resnet50(pretrained=False, quantize=False)
    else:
        raise ValueError(f"Invalid number of layers passed as argument -- {layers}")

def convert32(input_model, layers):

    # May need to change how we load the model for TorchScripts
    checkpoint = torch.load(input_model)
    model = get_resnet_model(checkpoint, layers)
    num_features = model.fc.in_features
    num_classes = 19
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.cuda()

    print("Loading weights...", end="")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("Complete!")

    x = torch.ones((1, 3, 224, 224)).cuda()

    print("Converting to TensorRT...", end="")
    sys.stdout.flush()
    model_trt_16 = torch2trt.torch2trt(model,[x], fp16_mode=True)
    torch.save(model_trt_16.state_dict(), "resnet18_trt_16.pth")
    print("<fp16> model saved...")
    sys.stdout.flush()
   # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
   # model_trt_32 = torch2trt.torch2trt(model, [x])
   # torch.save(model_trt_32.state_dict(), "resnet18_trt_32.pth")
   # print("<fp32> model saved...Complete!")
   # sys.stdout.flush()


def get_args():
    parser = argparse.ArgumentParser(
            prog="Pytorch to TensorRT Converter",
            description="Converts PyTorch Model to TensorRT Model." \
            )
    parser.add_argument(
        "--input_model", 
        type=pathlib.Path, 
        required=True, 
        help="Path to the input PyTorch model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        required=True, 
        help="Number of layers in the ResNet model (e.g., 18 for ResNet18, 50 for ResNet50)."
    )
    #parser.add_argument(
    #    "--output_dir",
    #    default="./",
    #    type=pathlib.Path,
    #    help="Path to save the converted TensorRT model (default: ./)."
    #)
    #parser.add_argument(
    #    "-fp16_mode",
    #    action="store_true",
    #    help="Enable FP16 mode for the TensorRT conversion."
    #)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    input_model = args.input_model
    layers = int(args.num_layers)
    #output_dir = args.output_dir
    convert32(input_model, layers)
