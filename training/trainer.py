import torch
import os
from glob import glob
from training.train_one_epoch import train_one_epoch
from diffusion.transition import sample_q_xt_given_x0
from visualisation.denoising import visualize_denoising
from training.metrics import denoising_accuracy

class Trainer:
    """
    High-level Trainer class for running multi-epoch D3PM training.
    
    Responsibilities:
      - Manage model, optimizer, dataloaders
      - Run train_one_epoch() for each epoch
      - Run optional validation
      - Save checkpoints
      - Track and return training statistics
    """

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        Qs,
        Qbar,
        T,
        device,
        val_loader=None,
        lambda_aux=0.001,
        ckpt_dir=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.Qs = Qs
        self.Qbar = Qbar
        self.T = T
        self.device = device
        self.lambda_aux = lambda_aux

        # Default checkpoint directory: training/checkpoints next to this file
        if ckpt_dir is None:
            ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Fix a sample for visualisation purposes
        x0_fixed = next(iter(train_loader))[0][0].unsqueeze(0)  # shape [1,1,28,28]
        self.x0_fixed = x0_fixed.to(device).long()

    # --------------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------------
    def save_checkpoint(self, epoch):
        if self.ckpt_dir is None:
            return

        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
        torch.save(ckpt, path)
        print(f"[Checkpoint saved]: {path}")

    # --------------------------------------------------------------
    # Load checkpoint (specific epoch or latest)
    # --------------------------------------------------------------
    def load_checkpoint(self, epoch=None, map_location=None, strict=True):
        if self.ckpt_dir is None:
            return None

        if epoch is None:
            # pick latest epoch_* file
            candidates = sorted(glob(os.path.join(self.ckpt_dir, "epoch_*.pt")))
            if not candidates:
                print("[Checkpoint load]: no checkpoints found")
                return None
            path = candidates[-1]
        else:
            path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt")
            if not os.path.exists(path):
                print(f"[Checkpoint load]: {path} not found")
                return None

        ckpt = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(ckpt["model"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        loaded_epoch = ckpt.get("epoch")
        print(f"[Checkpoint loaded]: {path} (epoch {loaded_epoch})")
        return loaded_epoch

    # --------------------------------------------------------------
    # Optional: Validation loop (no KL needed, only auxiliary CE)
    # --------------------------------------------------------------
    @torch.no_grad()
    def validate_one_epoch(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_ce = 0.0
        steps = 0

        for x0, _ in self.val_loader:
            x0 = x0.to(self.device).long()
            B = x0.shape[0]

            # random timesteps per sample
            t = torch.randint(1, self.T, (B,), device=self.device)

            # sample x_t
            xt = sample_q_xt_given_x0(x0, t, self.Qbar)

            # model prediction
            logits = self.model(xt, t)

            # compute -log p̃(x0 | x_t)
            p_tilde = torch.softmax(logits, dim=1)
            p_x0 = p_tilde.gather(1, x0)
            aux_ce = -(torch.log(p_x0 + 1e-20)).mean()

            total_ce += aux_ce.item()
            steps += 1

        return {"val_aux_ce": total_ce / steps}

    # --------------------------------------------------------------
    # Full training loop
    # --------------------------------------------------------------
    def train(
        self,
        num_epochs,
        print_every=1,
        validate=True,
        save_every=None,
    ):
        """
        Runs the full training loop over num_epochs.
        
        Returns:
            history: list of per-epoch stats
        """
        history = []

        for epoch in range(1, num_epochs + 1):
            # -----------------------------
            # Train for 1 epoch
            # -----------------------------
            stats = train_one_epoch(
                model=self.model,
                optimizer=self.optimizer,
                dataloader=self.train_loader,
                Qs=self.Qs,
                Qbar=self.Qbar,
                T=self.T,
                device=self.device,
                lambda_aux=self.lambda_aux,
            )
            
            # -----------------------------
            # Optional: validation
            # -----------------------------
            if validate and self.val_loader is not None:
                val_stats = self.validate_one_epoch()
                if val_stats:
                    stats.update(val_stats)

            history.append(stats)

            # -----------------------------
            # Logging
            # -----------------------------
            if print_every is not None and (epoch == 1 or epoch % print_every == 0):
                print(f"\nEpoch {epoch}/{num_epochs}")
                for k, v in stats.items():
                    print(f"  {k}: {v:.4f}")
                
                acc_mid = denoising_accuracy(self.model, self.Qbar,
                            self.train_loader, self.T,
                            self.device, t_eval=self.T//2, num_batches=5)
                print(f"  denoise_acc_t{self.T//2}: {acc_mid:.4f}")
                    
                visualize_denoising(
                    model=self.model,
                    Qbar=self.Qbar,
                    x0_sample=self.x0_fixed,
                    t=self.T // 2,
                    device=self.device
                )

            # -----------------------------
            # Checkpoint saving
            # -----------------------------
            if save_every is not None and epoch % save_every == 0:
                self.save_checkpoint(epoch)

        return history
