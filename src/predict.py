import torch
from PIL import Image
from src.model import SimpleCNN
from src.preprocess import get_inference_transforms

def predict(image_path, model_path):
    # 1. Load Model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. Load and Prep Image
    image = Image.open(image_path).convert('RGB')
    transform = get_inference_transforms()
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension

    # 3. Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, index = torch.max(probabilities, 0)
    
    return index.item(), confidence.item()