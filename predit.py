import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import pyttsx3

# Placeholder for your actual model loading code
def get_model(classes=7):
    # Placeholder for your model architecture
    model = torch.hub.load('pytorch/vision:v0.10.0','resnet50', pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=classes)
    return model

# Placeholder for your actual save path
save_path = "model_currency.pth"

# Load your model
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to make prediction
def predict_currency(model, image_path, device):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0)

    # model.eval()
    input_image = input_image.to(device, dtype=torch.float)
    with torch.no_grad():
        output = model(input_image)
    _, predicted_class = torch.max(output, 1)

    class_labels = ['Five Hundred rupees', 'One Hundred rupees', 'Two Hundred rupees', 'Ten rupees', 'Fifty rupees', 'Twenty rupees', 'Two Thousand rupees']
    predicted_label = class_labels[predicted_class.item()]

    return predicted_label

# Streamlit UI
def main():
    st.write("Hey Everyone this is Dinesh!!!  This is new Venture")
    st.title("Currency Recognition App")

    # Load model
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make prediction
        predicted_currency = predict_currency(model, uploaded_file, device)

        # Display prediction
        st.write(f"The predicted currency is: {predicted_currency}")

        # Convert prediction to speech
        engine = pyttsx3.init()
        engine.say(f"The predicted currency is: {predicted_currency}")
        engine.save_to_file(f"The predicted currency is: {predicted_currency}", 'predicted_currency.mp3')
        engine.runAndWait()

        # Play the speech
        st.audio('predicted_currency.mp3', format='audio/mp3')

if __name__ == '__main__':
    main()
