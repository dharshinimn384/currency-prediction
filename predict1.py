import streamlit as st
import cv2
import torch
from PIL import Image
from torchvision import transforms
from gtts import gTTS

# Placeholder for your actual model loading code
def get_model(classes=7):
    # Placeholder for your model architecture
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.DEFAULT')
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
def predict_currency(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(image).unsqueeze(0)

    input_image = input_image.to(device, dtype=torch.float)
    with torch.no_grad():
        output = model(input_image)
    _, predicted_class = torch.max(output, 1)

    class_labels = ['5Hundredrupees', '1Hundredrupees', '2Hundredrupees', 'Tenrupees', 'Fiftyrupees', 'Twenty rupees', '2Thousand rupees']
    predicted_label = class_labels[predicted_class.item()]

    return predicted_label

# Streamlit UI
def main():
    st.title("Currency Recognition")

    # Load model
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Webcam capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    st.write("Webcam is open. Press 'q' to capture an image.")

    captured_image = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            st.error("Error: Could not read webcam feed.")
            break

        # Display webcam feed
        st.image(frame, channels="BGR", use_column_width=True)

        # Check for key press to capture image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            captured_image = frame.copy()
            break

    # Turn off the webcam
    cap.release()

    # Convert the BGR image to RGB
    if captured_image is not None:
        rgb_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

        # Make prediction
        predicted_currency = predict_currency(model, Image.fromarray(rgb_image), device)

        # Display prediction
        st.write(f"The predicted currency is: {predicted_currency}")

        # Convert prediction to speech
        tts = gTTS(text=predicted_currency, lang='en')
        tts.save('predicted_currency.mp3')

        # Play the speech
        st.audio('predicted_currency.mp3', format='audio/mp3')

if __name__ == '__main__':
    main()
