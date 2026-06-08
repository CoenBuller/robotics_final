# train.py
import torch
import numpy as np
import librosa as lb
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SoundClassifier import AudioCNN
from torch import nn
from Training.AudioAugmentationPipelin import AudioAugmentationPipeline
from Training.config import AugmentConfig



def createLabels(audio_files: list[str], classes: list[str]):
    classes_dict = {c: i for i, c in enumerate(classes)}
    files = []
    labels = []

    for file in audio_files:
        c = file.split("_")[0]
        c = c.split("\\")[-1]

        files.append(file)
        labels.append(classes_dict[c])

    return files, labels


def _class_accuracy_table(class_correct: np.ndarray, class_total: np.ndarray, class_names: list[str]) -> str:
    """Return a compact per-class accuracy string for tqdm.write."""
    col_w = max(len(n) for n in class_names) + 2
    header = "  ".join(f"{n:>{col_w}}" for n in class_names)
    accs = []
    for correct, total in zip(class_correct, class_total):
        acc = (correct / total * 100) if total > 0 else 0.0
        accs.append(f"{acc:>{col_w}.1f}%")
    row = "  ".join(accs)
    return f"  {header}\n  {row}"


class AudioDataset(Dataset):
    def __init__(self, files, labels, audio_augmenter, hop=512, augment=True,
                 noise=True, pitch=True, volume=True, spec_aug=True,
                 timeshift=True, no_noise_labels: "set[int] | None" = None):
        self.X = files
        self.y = labels
        self.aa = audio_augmenter
        self.augment = augment
        self.hop = hop
        self.noise = noise
        self.pitch = pitch
        self.volume = volume
        self.spec_aug = spec_aug
        self.timeshift = timeshift
        self.no_noise_labels = no_noise_labels or set()

    def __getitem__(self, idx):
        label = self.y[idx]
        audio_data, sr = lb.load(path=self.X[idx], sr=self.aa.sr, duration=1)

        audio_len = len(audio_data)
        if audio_len < sr:
            audio_data = np.pad(audio_data, pad_width=(0, sr - audio_len))

        # Only apply augmentation on the training set, not validation
        x = self.aa.process(
            audio=audio_data,
            noise=self.augment and self.noise and (label not in self.no_noise_labels),
            volume=self.augment and self.volume,
            spec_aug=self.augment and self.spec_aug,
            time_shift=self.augment and self.timeshift,
            pitch=self.augment and self.pitch,
        )

        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), label

    def __len__(self):
        return len(self.X)


def _run_validation(model, loader, loss_fn, n_classes):
    """Run one full pass over the validation loader. Returns (avg_loss, correct, total)."""
    model.eval()
    val_loss = 0.0
    class_correct = np.zeros(n_classes, dtype=np.int64) # Counts correct predictions per class
    class_total   = np.zeros(n_classes, dtype=np.int64) # Counts total samples per class

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch) #
            val_loss += loss_fn(logits, y_batch).item() # Accumulate total loss for averaging
            preds = logits.argmax(dim=1) # Get predicted class indices
            for cls in range(n_classes):
                mask = y_batch == cls
                class_correct[cls] += (preds[mask] == cls).sum().item()
                class_total[cls]   += mask.sum().item()

    return val_loss / len(loader), class_correct, class_total


def train(
    files,
    labels,
    audio_processor,
    n_classes,
    class_names=None,
    epochs=100,
    lr=3e-3,
    no_noise_labels=None,
    val_split=0.1,
    patience=10,
    best_model_path="best_model2.pt",
):
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # ── Stratified train / val split ─────────────────────────────────────────
    files  = np.array(files)
    labels = np.array(labels)

    train_files, val_files, train_labels, val_labels = train_test_split(
        files, labels,
        test_size=val_split,
        stratify=labels,        # keeps class ratios equal in both splits
        random_state=42,
    )

    tqdm.write(f"Train samples: {len(train_files)}  |  Val samples: {len(val_files)}")
    for cls_idx, name in enumerate(class_names):
        n_train = (train_labels == cls_idx).sum()
        n_val   = (val_labels   == cls_idx).sum()
        tqdm.write(f"  {name}: {n_train} train / {n_val} val")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = AudioDataset(
        train_files.tolist(), train_labels.tolist(),
        audio_processor, augment=True, no_noise_labels=no_noise_labels,
    )
    val_dataset = AudioDataset(
        val_files.tolist(), val_labels.tolist(),
        audio_processor, augment=False,   # no augmentation on val
    )

    # Weighted sampler on train only to counter class imbalance
    class_counts   = np.bincount(train_labels, minlength=n_classes)
    class_weights  = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [float(class_weights[l]) for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8,
                              sampler=sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=64, num_workers=8,
                              shuffle=False)

    steps_per_epoch = len(train_loader)

    #  Model / optimiser / scheduler 
    model   = AudioCNN(n_classes=n_classes)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    total_warmup_steps = max(1, int(epochs * 0.1)) * steps_per_epoch
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.1,       # begin at lr * 0.1
        end_factor=1.0,         # reach full lr at end of warmup
        total_iters=total_warmup_steps,
    )
    # After warm-up, val loss drives LR reductions
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,             # halve LR on plateau
        patience=5,             # wait 5 epochs before reducing
        min_lr=1e-6,
    )
    global_step = 0             # counts total batches seen across all epochs

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0)

    #  Early-stopping state 
    best_val_loss    = float("inf")
    epochs_no_improve = 0

    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0)

    for epoch in epoch_bar:
        #  Training pass 
        model.train()
        total_loss    = 0.0
        class_correct = np.zeros(n_classes, dtype=np.int64)
        class_total   = np.zeros(n_classes, dtype=np.int64)

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:>4}/{epochs}",
            position=1,
            leave=False,
            total=steps_per_epoch,
        )

        for batch_idx, (X_batch, y_batch) in enumerate(batch_bar):
            opt.zero_grad()
            logits = model(X_batch)
            loss   = loss_fn(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # LinearLR steps every batch; becomes a no-op after total_warmup_steps
            if global_step < total_warmup_steps:
                warmup_sched.step()
            global_step += 1

            total_loss += loss.item() 
            preds = logits.argmax(dim=1)

            for cls in range(n_classes):
                mask = y_batch == cls
                class_correct[cls] += (preds[mask] == cls).sum().item()
                class_total[cls]   += mask.sum().item()

            batch_bar.set_postfix(
                batch=f"{batch_idx + 1}/{steps_per_epoch}",
                loss=f"{loss.item():.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

        avg_train_loss = total_loss / steps_per_epoch
        train_acc      = class_correct.sum() / class_total.sum()

        #  Validation pass 
        avg_val_loss, val_correct, val_total = _run_validation(
            model, val_loader, loss_fn, n_classes
        )
        val_acc = val_correct.sum() / val_total.sum()

        # ReduceLROnPlateau steps on val loss (after warm-up)
        if global_step > total_warmup_steps:
            plateau_sched.step(avg_val_loss)

        #  Logging 
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            val_acc=f"{val_acc * 100:.1f}%",
        )

        val_table = _class_accuracy_table(val_correct, val_total, class_names)
        tqdm.write(
            f"\nEpoch {epoch + 1}"
            f" — train_loss: {avg_train_loss:.4f}  train_acc: {train_acc * 100:.1f}%"
            f" — val_loss: {avg_val_loss:.4f}  val_acc: {val_acc * 100:.1f}%"
            f"  lr: {opt.param_groups[0]['lr']:.2e}"
            f"\n  val per-class:\n{val_table}\n"
        )

        #  Early stopping & best model saving 
        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"  ✓ New best val_loss={best_val_loss:.4f} — model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            tqdm.write(f"  No improvement for {epochs_no_improve}/{patience} epochs")

            if epochs_no_improve >= patience:
                tqdm.write(
                    f"\n⚑ Early stopping at epoch {epoch + 1}. "
                    f"Best val_loss: {best_val_loss:.4f}. "
                    f"Loading best weights from {best_model_path}."
                )
                model.load_state_dict(torch.load(best_model_path))
                break

    return model


if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    class_folders = os.listdir(data_folder)

    audio_files = []
    labels      = []

    # Read all files and label them by their folder name
    for label, folder in tqdm(enumerate(class_folders)):
        p     = os.path.join(data_folder, folder)
        files = os.listdir(p)
        for file in files:
            fp = os.path.join(p, file)
            audio_files.append(fp)
            labels.append(label)

    print(f"Classes: {class_folders}")
    print(np.unique(labels))

    cfg = AugmentConfig(
        noise_prob=0.6,        noise_snr_range=(5, 30),
        pitch_shift_prob=0.5,  pitch_shift_range=(-4, 4),
        time_shift_prob=0.5,   time_shift_range=(-0.4, 0.1),
        volume_scale_prob=0.4, volume_gain_range=(0.6, 1.2),
        spec_augment_prob=0.2,
        n_freq_masks=8,
        n_time_masks=8,
    )


    ap = AudioAugmentationPipeline(config=cfg, n_mels=62, hop=512, n_fft=512, sr=15_872)

    model = train(
        files=audio_files,
        labels=labels,
        audio_processor=ap,
        n_classes=len(class_folders),
        class_names=class_folders,
        epochs=80,
        val_split=0.1,          # 10 % held out for validation
        patience=10,            # stop after 10 epochs without val_loss improvement
        best_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", input("Enter filename to save best model (e.g. best_model.pt): ")),
    )