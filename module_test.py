import torch
import torch.nn as nn
from tqdm import tqdm
from config.config import get_config_regression
from data_loader import MMDataLoader
from models import AMIO
from utils import *

def test(model_name, dataset_name, featurePath):
    """Main testing function for multimodal regression model.

    Args:
        model_name (str): Name of the model architecture
        dataset_name (str): Name of the dataset
        featurePath (str): Path to preprocessed feature file
    """
    # Set a fixed seed for reproducibility
    set_seed(42)

    # Load configuration from JSON file
    config_file = r"config/config_pretrained.json"
    args = get_config_regression(model_name, dataset_name, config_file)

    # Update arguments with feature path and device selection
    args["featurePath"] = featurePath
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize data loader with batch size 1
    dataloader = MMDataLoader(args, 1)

    # Initialize model and move to appropriate device
    model = AMIO(args).to(args['device'])

    # Load pretrained weights
    model_path = fr"pretrained_model/{model_name}_pretrained_model.pth"
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(args['device'])

    # Execute testing procedure
    do_test(args, model_name, dataset_name, model, dataloader)


def do_test(args, model_name, dataset_name, model, dataloader):
    """Execute model testing with evaluation metrics.

    Args:
        args: Configuration dictionary
        model_name: Name of model architecture
        dataset_name: Name of dataset
        model: Loaded model instance
        dataloader: Initialized data loader
    """
    # Initialize loss function (Mean Absolute Error)
    criterion = nn.L1Loss()

    # Get appropriate metrics for the dataset
    metrics = MetricsTop(args.test_mode).getMetrics(args.dataset_name)

    # Set model to evaluation mode
    model.eval()

    # Initialize containers for predictions and ground truth
    y_pred, y_true = [], []  # For final aggregated results
    info_s = []  # For storing meta information

    eval_loss = 0.0  # Accumulated loss

    # Disable gradient calculation for testing
    with torch.no_grad():
        # Process each batch with progress bar
        for batch_data in tqdm(dataloader, desc="Processing", ncols=100):
            # Move data to appropriate device
            vision = batch_data['vision'].to(args.device)
            audio = batch_data['audio'].to(args.device)
            text = batch_data['text'].to(args.device)

            # Get labels and reshape
            labels = batch_data['labels']['M'].to(args.device)
            labels = labels.view(-1, 1)  # Ensure proper shape

            # Store meta information
            info = batch_data["meta_info"]
            info_s += info  # list of strings

            # Forward pass
            outputs = model(text, audio, vision)['M']

            # Calculate loss
            loss = criterion(outputs, labels)
            eval_loss += loss.item()  # Accumulate batch loss

            # Store tensors for final metrics calculation
            y_pred.append(outputs.cpu())
            y_true.append(labels.cpu())

        # Calculate average loss
        train_loss = eval_loss / len(dataloader)

        # Concatenate all batch results
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        # Store numpy arrays
        y_pred.append(pred.cpu().numpy())
        y_true.append(true.cpu().numpy())

    # Calculate evaluation metrics
    eval_results = metrics(pred, true)

    # Print formatted results
    print("The results are as follows:")
    print(eval_results)
