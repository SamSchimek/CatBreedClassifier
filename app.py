import gradio as gr
from fastai.vision.all import load_learner, PILImage

# Load the trained model
learn = load_learner('model.pkl')

# Define the prediction function
labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Pet Breed Classifier",
    description="A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces.",
    examples=["example1.jpg", "example2.jpg", "example3.jpg"]  # Replace with actual filenames
)

# Launch the interface
interface.launch()
