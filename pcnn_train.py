import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths

NUM_CLASSES = 4

def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode = 'training'):
    if mode == 'training':
        model.train()
    else:
        model.eval()
        
    deno = args.batch_size * np.prod(args.obs) * np.log(2.)        
    loss_tracker = mean_tracker()
    correct = 0
    total = 0
    batch_losses = []

    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories, _ = item  # Changed to match evaluation code
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]  # Convert categories to labels
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        
        if mode == 'training':
            # During training, we only compute loss for the true labels
            model_output = model(model_input, original_label)  # Use original_label instead of labels
            loss = loss_op(model_input, model_output)
            loss_tracker.update(loss.item()/deno)
            batch_losses.append(loss.item()/deno)
            
            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            # During evaluation, we need to try all possible class labels
            with torch.no_grad():
                B = model_input.shape[0]
                class_losses = torch.zeros((NUM_CLASSES, B), device=device)
                
                # Try each possible class label
                for class_label in range(NUM_CLASSES):
                    class_cond = torch.full((B,), class_label, device=device)
                    predictions = model(model_input, class_cond)
                    loss = loss_op(model_input, predictions)
                    class_losses[class_label] = loss
                
                # Get predictions and update accuracy
                predicted = torch.argmin(class_losses, dim=0)
                total += original_label.size(0)
                correct += (predicted == original_label).sum().item()
                
                # Update loss tracker with the loss for true labels
                true_label_losses = torch.gather(class_losses, 0, original_label.unsqueeze(0))[0]
                loss_tracker.update(true_label_losses.mean().item()/deno)
                batch_losses.append(true_label_losses.mean().item()/deno)
    
    accuracy = 100 * correct / total if total > 0 else 0
    
    if args.en_wandb:
        wandb.log({mode + "-Average-BPD" : loss_tracker.get_mean()})
        wandb.log({mode + "-epoch": epoch})
        wandb.log({mode + "-accuracy": accuracy})
        # Log batch losses
        wandb.log({mode + "-batch_losses": wandb.Histogram(batch_losses)})
        
    return accuracy, loss_tracker.get_mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-w', '--en_wandb', type=bool, default=False,
                            help='Enable wandb logging')
    parser.add_argument('-t', '--tag', type=str, default='default',
                            help='Tag for this run')
    
    # sampling
    parser.add_argument('-c', '--sampling_interval', type=int, default=5,
                        help='sampling interval')
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-sd', '--sample_dir',  type=str, default='samples',
                        help='Location for saving samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cpen455', help='Can be either cifar|mnist|cpen455')
    parser.add_argument('-st', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--obs', type=tuple, default=(3, 32, 32),
                        help='Observation shape')
    
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0001, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size during training per GPU')
    parser.add_argument('-sb', '--sample_batch_size', type=int, default=32,
                        help='Batch size during sampling per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    check_dir_and_create(args.save_dir)
    
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model_name = 'pcnn_' + args.dataset + "_"
    model_path = args.save_dir + '/'
    if args.load_params is not None:
        model_name = model_name + 'load_model'
        model_path = model_path + model_name + '/'
    else:
        model_name = model_name + 'from_scratch'
        model_path = model_path + model_name + '/'
    
    job_name = "PCNN_Training_" + "dataset:" + args.dataset + "_" + args.tag
    
    if args.en_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set entity to specify your username or team name
            # entity="qihangz-work",
            # set the wandb project where this run will be logged
            project="CPEN455HW",
            # group=Group Name
            name=job_name,
        )
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wandb.config.update(args)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Reminder: if you have patience to read code line by line, you should notice this comment. here is the reason why we set num_workers to 0:
    #In order to avoid pickling errors with the dataset on different machines, we set num_workers to 0.
    #If you are using ubuntu/linux/colab, and find that loading data is too slow, you can set num_workers to 1 or even bigger.
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':True}

    # set data
    if "mnist" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), rescaling, replicate_color_channel])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                            train=True, transform=ds_transforms), batch_size=args.batch_size, 
                                shuffle=True, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    elif "cifar" in args.dataset:
        ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
        if args.dataset == "cifar10":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        elif args.dataset == "cifar100":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            raise Exception('{} dataset not in {cifar10, cifar100}'.format(args.dataset))
    
    elif "cpen455" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
        ds_transforms_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            transforms.Resize((32, 32)),
            rescaling
        ])

        train_loader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'train', 
                                                                  transform=ds_transforms_train), 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
        # test_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
        #                                                           mode = 'test', 
        #                                                           transform=ds_transforms), 
        #                                            batch_size=args.batch_size, 
        #                                            shuffle=True, 
        #                                            **kwargs)
        val_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'validation', 
                                                                  transform=ds_transforms), 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
    else:
        raise Exception('{} dataset not in {mnist, cifar, cpen455}'.format(args.dataset))
    
    args.obs = (3, 32, 32)
    input_channels = args.obs[0]
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    model = model.to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    
    # Add early stopping parameters
    best_val_accuracy = 0
    patience = 15  # Increased patience
    patience_counter = 0
    best_epoch = 0
    
    for epoch in tqdm(range(args.max_epochs)):
        train_accuracy, train_loss = train_or_test(model = model, 
                      data_loader = train_loader, 
                      optimizer = optimizer, 
                      loss_op = loss_op, 
                      device = device, 
                      args = args, 
                      epoch = epoch, 
                      mode = 'training')
        
        val_accuracy, val_loss = train_or_test(model = model,
                      data_loader = val_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'val')
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_accuracy)
        
        # Log accuracy and loss every epoch
        print(f'Epoch {epoch}:')
        print(f'Train - Accuracy: {train_accuracy:.2f}%, Loss: {train_loss:.4f}')
        print(f'Val - Accuracy: {val_accuracy:.2f}%, Loss: {val_loss:.4f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), 'models/best_{}_{}.pth'.format(model_name, epoch))
            print(f'New best model saved with validation accuracy: {val_accuracy:.2f}%')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement')
            print(f'Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch}')
            break
        
        if epoch % args.sampling_interval == 0:
            print('......sampling......')
            
            
            for label in my_bidict.values():
                print(f"Generating images for label: {label}")
                class_cond = torch.full((args.sample_batch_size,), label).to(next(model.parameters()).device)
                sample_t = sample(model, args.sample_batch_size, args.obs, sample_op, class_cond=class_cond)
                sample_t = rescaling_inv(sample_t)
                save_images(sample_t, args.sample_dir, label=label)
                sample_result = wandb.Image(sample_t, caption="epoch {} - label {}".format(epoch, label))
            
            gen_data_dir = args.sample_dir
            ref_data_dir = args.data_dir +'/test'
            paths = [gen_data_dir, ref_data_dir]
            try:
                fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
                print("Dimension {:d} works! fid score: {}".format(192, fid_score))
            except:
                print("Dimension {:d} fails!".format(192))
                
            if args.en_wandb:
                wandb.log({"samples": sample_result,
                            "FID": fid_score})
        
        if (epoch + 1) % args.save_interval == 0: 
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
