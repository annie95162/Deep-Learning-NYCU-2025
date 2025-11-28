import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import random
#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers(args)
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_epoch(self, dataloader, config, epoch_idx):
        progress = tqdm(dataloader)
        epoch_loss = torch.tensor(0.0, device=config.device)
        step_count = 0

        for step, images in enumerate(progress):
            images = images.to(config.device)
            preds, targets = self.model(images)

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets.view(-1))
            epoch_loss += loss
            step_count += 1

            loss.backward()

            if step % config.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()

            progress.set_description(f"[Train] Epoch {epoch_idx}")
            progress.set_postfix({
                'LR': f"{self.scheduler.get_last_lr()[0]:.6f}",
                'Loss': f"{(epoch_loss.item() / step_count):.4f}"
            }, refresh=False)

        self.scheduler.step()
        return epoch_loss.item() / step_count

    @torch.no_grad()
    def validate_epoch(self, dataloader, config, epoch_idx):
        progress = tqdm(dataloader)
        val_loss = torch.tensor(0.0, device=config.device)
        count = 0

        for step, images in enumerate(progress):
            images = images.to(config.device)
            preds, targets = self.model(images)

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets.view(-1))
            val_loss += loss
            count += 1

            progress.set_description(f"[Val] Epoch {epoch_idx}")
            progress.set_postfix({'Loss': f"{(val_loss.item() / count):.4f}"}, refresh=False)

        return val_loss.item() / count

    def configure_optimizers(self, config):
        opt = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2, 10], gamma=0.1)
        return opt, sched

    def save_checkpoint(self, file_path):
        torch.save({
            'transformer_state': self.model.transformer.state_dict(),
            'optimizer_state': self.optim.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }, file_path)
        print(f"Checkpoint saved to: {file_path}")

    def load_checkpoint(self, config):
        if config.load_path is not None:
            data = torch.load(config.load_path)
            self.model.transformer.load_state_dict(data['transformer_state'])
            self.optim.load_state_dict(data['optimizer_state'])
            self.scheduler.load_state_dict(data['scheduler_state'])

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./transformer_checkpoints', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0, help='Learning rate.')
    parser.add_argument('--load_path', type=str, default=None, help='the path to load ckpt')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    # TODO2 step1-5 (revised version)

    train_transformer.load_checkpoint(args)

    lowest_train_loss = float('inf')
    lowest_val_loss = float('inf')

    for epoch_idx in range(args.start_from_epoch + 1, args.epochs + 1):
        avg_train_loss = train_transformer.train_epoch(train_loader, args, epoch_idx)
        avg_val_loss = train_transformer.validate_epoch(val_loader, args, epoch_idx)

        # Save checkpoint every N epochs
        if epoch_idx % args.save_per_epoch == 0:
            train_transformer.save_checkpoint(f"{args.checkpoint_path}/epoch_{epoch_idx}.pt")

        # Save best train loss model
        if avg_train_loss < lowest_train_loss:
            lowest_train_loss = avg_train_loss
            train_transformer.save_checkpoint(f"{args.checkpoint_path}/best_train.pt")

        # Save best validation loss model
        if avg_val_loss < lowest_val_loss:
            lowest_val_loss = avg_val_loss
            train_transformer.save_checkpoint(f"{args.checkpoint_path}/best_val.pt")
