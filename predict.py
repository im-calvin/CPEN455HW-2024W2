import os
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import Compose, Resize
from dataset import CPEN455Dataset, rescaling, my_bidict
from model import PixelCNN
from utils import discretized_mix_logistic_loss
from tqdm import tqdm

NUM_CLASSES = len(my_bidict)


def get_label(model, model_input, device):
    """
    Classify images by finding which class label minimizes the generation loss.

    Args:
        model: The trained PixelCNN model
        model_input: Input images of shape (batch_size, channels, height, width)
        device: Device to run computations on

    Returns:
        Tensor of predicted class labels for each image in the batch
    """
    B, _, _, _ = model_input.shape

    # Log likelihood tensor move to device
    loss_from_log_likelihood = torch.zeros((NUM_CLASSES, B))
    loss_from_log_likelihood = loss_from_log_likelihood.to(device)

    for possible_class in my_bidict.values():
        # Predicts based on the current label
        answer = model(model_input, torch.full((B,), possible_class, device=device))

        # Calculate loss between predictions and actual input
        loss_from_log_likelihood[possible_class, :] = discretized_mix_logistic_loss(
            model_input, answer, training=False
        )

    # For each image, find the class that gave the lowest loss
    predicted_classes = torch.argmin(loss_from_log_likelihood, dim=0)

    return predicted_classes, loss_from_log_likelihood


def main():
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelCNN(
        nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=10
    ).to(device)

    # Load trained weights
    model_path = os.path.join(
        os.path.dirname(__file__), "models/conditional_pixelcnn.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model parameters loaded")

    # Initialize dataset and dataloader
    transform = Compose([Resize((32, 32)), rescaling])
    test_dataset = CPEN455Dataset(root_dir="./data", mode="test", transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    # Initialize results list
    results = []
    logits = []

    # Predict on test images
    with torch.no_grad():
        for images, _, _ in tqdm(test_loader):
            images = images.to(device)
            # Get predictions
            predicted_classes, logit = get_label(model, images, device)
            results.extend(predicted_classes.cpu().numpy())
            logits.append(logit.T.cpu().detach().numpy())

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

    # Save logits
    logits_arr = np.concatenate(logits, axis=0)
    np.save("logits.npy", logits_arr)
    print("Logits saved to logits.npy")


if __name__ == "__main__":
    main()
