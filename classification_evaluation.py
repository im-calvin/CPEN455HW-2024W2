"""
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
"""

from torchvision import datasets, transforms
from utils import *
from model import *
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv

NUM_CLASSES = len(my_bidict)


# TODO: Begin of your code
# def get_label(model, model_input, device):
#     # Write your code here, replace the random classifier with your trained model
#     # and return the predicted label, which is a tensor of shape (batch_size,)
#     answer = model(model_input, device)
#     return answer

# End of your code


# TODO: Begin of your code
def get_label(model, model_input, device):
    # model_input shape is (batch_size, 3, 32, 32)
    model_input = model_input.to(device)
    batch_size = model_input.shape[0]

    # Compute the log-likelihood for each class
    log_probs = []
    with torch.no_grad():
        for class_idx in range(NUM_CLASSES):
            # create a one-hot vector for the current class
            class_condition = torch.zeros(batch_size, NUM_CLASSES).to(device)
            class_condition[:, class_idx] = 1

            # forward pass through the model, conditioned on class_condition
            output = model(model_input, sample=False, class_cond=class_condition)

            # compute the log-likelihood of model_input given the current class
            ll = discretized_mix_logistic_loss(model_input, output, sum_all=False)
            ll = -ll.view(batch_size, -1).sum(dim=1)  # negative loss is log-probability

            log_probs.append(ll.unsqueeze(1))  # shape (batch_size, 1)

    # Stack the log probabilities to shape (batch_size, num_classes)
    log_probs = torch.cat(log_probs, dim=1)  # (batch_size, NUM_CLASSES)

    # predict the label: the class with the highest log_prob
    predicted_label = torch.argmax(log_probs, dim=1)

    return predicted_label  # tensor shape: (batch_size,)


# End of your code


def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])

    return acc_tracker.get_ratio()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--data_dir", type=str, default="data", help="Location for the dataset"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="validation", help="Mode for the dataset"
    )

    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {"num_workers": 0, "pin_memory": True, "drop_last": False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )

    # TODO:Begin of your code
    model = PixelCNN(
        nr_resnet=1,
        nr_filters=40,
        input_channels=3,
        nr_logistic_mix=5,
        num_classes=NUM_CLASSES,
    ).to(device)
    # End of your code

    model = model.to(device)
    # Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    # You should save your model to this path
    model_path = os.path.join(
        os.path.dirname(__file__), "models/conditional_pixelcnn.pth"
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("model parameters loaded")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()

    acc = classifier(model=model, data_loader=dataloader, device=device)
    print(f"Accuracy: {acc}")
