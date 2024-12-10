import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dataset import MIDIDataset
from .vae import MusicVAE

class Trainer:
    def __init__(
        self,
        model: MusicVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        checkpoint_dir: str = 'models/checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
    
    def compute_loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        genre_pred: torch.Tensor,
        genre_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Genre classification loss
        genre_loss = F.cross_entropy(genre_pred, genre_true)
        
        # Total loss
        total_loss = recon_loss + 0.01 * kl_loss + genre_loss
        
        # Compute accuracy
        pred = genre_pred.argmax(dim=1)
        accuracy = (pred == genre_true).float().mean().item()
        
        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'genre_loss': genre_loss.item(),
            'accuracy': accuracy
        }
        
        return total_loss, metrics
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_metrics = {'loss': 0, 'accuracy': 0}
        
        for batch_idx, (data, genre) in enumerate(tqdm(self.train_loader)):
            data = data.to(self.device)
            genre = genre.to(self.device)
            
            self.optimizer.zero_grad()
            recon, mu, log_var, genre_pred = self.model(data)
            
            loss, metrics = self.compute_loss(recon, data, mu, log_var, genre_pred, genre)
            
            loss.backward()
            self.optimizer.step()
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
        # Average metrics
        num_batches = len(self.train_loader)
        return {k: v / num_batches for k, v in total_metrics.items()}
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_metrics = {'loss': 0, 'accuracy': 0}
        
        with torch.no_grad():
            for data, genre in self.val_loader:
                data = data.to(self.device)
                genre = genre.to(self.device)
                
                recon, mu, log_var, genre_pred = self.model(data)
                _, metrics = self.compute_loss(recon, data, mu, log_var, genre_pred, genre)
                
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
        
        # Average metrics
        num_batches = len(self.val_loader)
        return {k: v / num_batches for k, v in total_metrics.items()}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pt')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
    
    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(epoch, val_metrics, is_best)

def train_model(
    data_dir: str = 'data/midi',
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    num_workers: int = 4
):
    # Create datasets
    train_dataset = MIDIDataset(data_dir, 'train', min_samples=100)
    val_dataset = MIDIDataset(data_dir, 'val')

    # Create samplers
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = MusicVAE(input_dim=29, 
                     hidden_dim=1024, 
                    latent_dim=512,
                    num_genres=len(train_dataset.genre_mapping))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate
    )
    
    # Train model
    trainer.train(num_epochs)

if __name__ == "__main__":
    train_model()