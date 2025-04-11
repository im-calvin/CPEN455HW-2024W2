import os
import torch
import pandas as pd
from torchvision.transforms import Compose, Resize
from dataset import CPEN455Dataset, rescaling
from model import PixelCNN
from tqdm import tqdm


def main():
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelCNN().to(device)

    # Load trained weights
    model_path = os.path.join(
        os.path.dirname(__file__), "models/conditional_pixelcnn.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize dataset and dataloader
    transform = Compose([Resize((32, 32)), rescaling])

    test_dataset = CPEN455Dataset(root_dir="./data", mode="test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize results list
    results = []

    # Predict on test images
    with torch.no_grad():
        for images, _, _ in tqdm(test_loader):
            images = images.to(device)
            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())

    # Create submission DataFrame
    submission = pd.DataFrame(
        {
            "path": [os.path.basename(path) for path, _ in test_dataset.samples],
            "label": results,
        }
    )

    # Save to CSV
    submission.to_csv("submission.csv", index=False)
    print("Predictions saved to submission.csv")


if __name__ == "__main__":
    main()
