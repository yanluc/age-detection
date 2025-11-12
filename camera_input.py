import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import effnetv2_xs

def main():
    CHECKPOINT_PATH = f'data/checkpoints/best_model_effnetv2_xs.pth'
    INPUT_SIZE = 200
    model = effnetv2_xs(num_classes=1)

    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    data_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    estimated_age = None
    last_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        last_face = frame

        if estimated_age is not None:
            cv2.putText(frame, f"Estimated Age: {estimated_age:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Age Estimation - Press SPACE to estimate, Q to quit', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if last_face is not None:
                try:
                    face_pil = Image.fromarray(cv2.cvtColor(last_face, cv2.COLOR_BGR2RGB))
                    
                    input_tensor = data_transform(face_pil)
                    input_batch = input_tensor.unsqueeze(0)

                    with torch.no_grad():
                        output = model(input_batch)
                        estimated_age = output.item()
                    
                    print(f"Age estimated: {estimated_age:.1f}")

                except Exception as e:
                    print(f"An error occurred during age estimation: {e}")
            else:
                print("No face detected to estimate age.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
