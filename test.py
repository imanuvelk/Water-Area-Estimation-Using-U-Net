import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
import segmentation_models_pytorch as smp

# CONFIGURATION
MODEL_PATH = 'C:\\Users\\IMANUVEL K\\OneDrive\\Desktop\\AIML NOTES\\BDA Project\\TEST\\best_model.pt'
CALIBRATION_IMAGE_PATH = 'C:\\Users\\IMANUVEL K\\OneDrive\\Desktop\\AIML NOTES\\BDA Project\\TEST\\lake_image.jpg'
TEST_IMAGE_PATH = 'C:\\Users\\IMANUVEL K\\OneDrive\\Desktop\\AIML NOTES\\BDA Project\\TEST\\test_image4.png'
IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5
GROUND_TRUTH_AREA_KM2 = 25.51  # Known area of calibration image
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# MODEL DEFINITION
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.arc = smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.arc(x)

# IMAGE PREPROCESSING
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    aug = A.Resize(*IMAGE_SIZE)
    image = aug(image=image)['image']
    image = np.transpose(image, (2, 0, 1)) / 255.0
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# CALIBRATION STEP
def calibrate_pixel_area(model, image_path, ground_truth_area_km2):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor))
    binary_mask = (output > THRESHOLD).float()
    predicted_pixels = binary_mask.sum().item()

    pixel_area_km2 = ground_truth_area_km2 / predicted_pixels
    pixel_area_m2 = pixel_area_km2 * 1e6
    print(f"🔧 Calibrated pixel area: {pixel_area_m2:.2f} m²")
    return pixel_area_m2, binary_mask.squeeze().cpu().numpy()

# AREA ESTIMATION STEP
def estimate_area(model, image_path, pixel_area_m2):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor))
    binary_mask = (output > THRESHOLD).float()
    water_pixels = binary_mask.sum().item()

    area_m2 = water_pixels * pixel_area_m2
    area_ha = area_m2 / 10000
    area_km2 = area_m2 / 1e6
    print(f"\n🌊 Estimated water area from test image:")
    print(f"   🔹 {area_m2:,.2f} m²")
    print(f"   🔹 {area_ha:,.2f} hectares")
    print(f"   🔹 {area_km2:,.4f} km²")

    return binary_mask.squeeze().cpu().numpy()

# VISUALIZATION
def visualize_results(image_path, predicted_mask):
    image = Image.open(image_path).resize(IMAGE_SIZE)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='Blues')
    plt.title("Predicted Water Mask")

    plt.tight_layout()
    plt.show()

# MAIN
def main():
    print("🔁 Loading model...")
    model = SegmentationModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("🧪 Calibrating using known image...")
    pixel_area_m2, _ = calibrate_pixel_area(model, CALIBRATION_IMAGE_PATH, GROUND_TRUTH_AREA_KM2)

    print("🧮 Estimating area for test image...")
    predicted_mask = estimate_area(model, TEST_IMAGE_PATH, pixel_area_m2)

    visualize_results(TEST_IMAGE_PATH, predicted_mask)

if __name__ == "__main__":
    main()
