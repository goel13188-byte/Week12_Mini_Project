import gradio as gr
from src.predict import predict

# Define your labels (e.g., CIFAR-10 classes)
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def inference_interface(img):
    # Save temporary image to disk for the predict function
    img.save("temp.jpg")
    idx, conf = predict("temp.jpg", "models/best_model.pth")
    return {labels[idx]: float(conf)}

demo = gr.Interface(
    fn=inference_interface,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="My First Image Classifier"
)

if __name__ == "__main__":
    demo.launch(share=True)