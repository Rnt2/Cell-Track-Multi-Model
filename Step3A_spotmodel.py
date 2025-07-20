# BiLSTM with Attention for Variable-Length Input
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from Config import *  # Assuming Config.py contains necessary configurations
# ===== Global Config =====
DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
MODEL_PATH = f"{MODEL_DIR}/model_best_{SEQ_LEN}.pth"

# ===== Load Data =====
data = np.load(DATA_PATH)
X, y = data["X"], data["y"]

# Transpose if needed
if X.shape[1] == FEATURE_LEN and X.shape[2] == SEQ_LEN:
    X = np.transpose(X, (0, 2, 1))

X = torch.tensor(X, dtype=torch.float32)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded = torch.tensor(y_encoded, dtype=torch.long)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

# ===== Model =====
class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        weights = self.dropout(weights)
        context = torch.sum(weights.unsqueeze(-1) * lstm_out, dim=1)
        return context, weights

class BiLSTMAttnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim*2, dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attn(lstm_out)
        return self.fc(context), attn_weights

#128, 32
#64, 32
#64, 16

model = BiLSTMAttnModel(input_dim=FEATURE_LEN, hidden_dim=128, output_dim=3, dropout=0.5)
class_weights = torch.tensor([1.0 / (y_encoded == i).sum().item() for i in range(3)])
class_weights = class_weights / class_weights.sum()
#class_weights[2] = 1.0
print(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10)

# ===== Training =====
def train_model():
    best_acc, early_stop = 0, 0
    best_model = None
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, 250):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            out, _ = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                out, _ = model(Xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        val_loss /= len(val_loader)
        acc = correct / total
        print(scheduler.get_last_lr())
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(acc)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Acc={acc:.4f}")

        if acc > best_acc:
            best_acc, best_model = acc, model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 1002:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), MODEL_PATH)
    np.savez(f"{SEQ_RESULT_DIR}/training_logs.npz", train_losses=train_losses, val_losses=val_losses, val_accuracies=val_accs)

    return train_losses, val_losses, val_accs

def pca_plot(ctx_vecs):
    pca = PCA(n_components=2)
    ctx_pca = pca.fit_transform(ctx_vecs)
    sns.scatterplot(x=ctx_pca[:,0], y=ctx_pca[:,1],
                    hue=le.inverse_transform(y_encoded.numpy()))
    plt.title("PCA of Attention Outputs")
    plt.savefig(f"{SEQ_RESULT_DIR}/pca_attention.png")
    plt.close()
    
def tsne_plot(ctx_vecs, per_para=30):
    tsne = TSNE(perplexity=per_para, random_state=42)
    ctx_tsne = tsne.fit_transform(ctx_vecs)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=ctx_tsne[:,0], y=ctx_tsne[:,1],
                    hue=le.inverse_transform(y_encoded.numpy()), alpha=0.7)
    plt.title("t-SNE of Attention Vectors")
    plt.tight_layout()
    plt.savefig(f"{SEQ_RESULT_DIR}/tsne_attention_{per_para}.png")
    plt.close()

def umap_plot(ctx_vecs, nei_para=15):
    reducer = umap.UMAP(n_neighbors=nei_para, min_dist=0.5)
    ctx_umap = reducer.fit_transform(ctx_vecs)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=ctx_umap[:,0], y=ctx_umap[:,1],
                    hue=le.inverse_transform(y_encoded.numpy()), alpha=0.7)
    plt.title("UMAP of Attention Vectors")
    plt.tight_layout()
    plt.savefig(f"{SEQ_RESULT_DIR}/umap_attention_{nei_para}.png")
    plt.close()


# ===== Main =====
if __name__ == "__main__":
    import random
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    if os.path.exists(MODEL_PATH) and False:
        print("Loading existing model...")
        model.load_state_dict(torch.load(MODEL_PATH,weights_only=True))
        logs = np.load(f"{SEQ_RESULT_DIR}/training_logs.npz")
        train_losses, val_losses, val_accs = logs['train_losses'], logs['val_losses'], logs['val_accuracies']
    else:
        print("Training model from scratch...")
        train_losses, val_losses, val_accs = train_model()

    # Plot training curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{SEQ_RESULT_DIR}/loss_curve.png")
    plt.close()

    plt.plot(np.array(val_accs)*100)
    plt.title("Validation Accuracy (%)")
    plt.savefig(f"{SEQ_RESULT_DIR}/val_accuracy.png")
    plt.close()

    # Evaluate
    model.eval()
    y_true, y_pred, probs, attns = [], [], [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            out, att = model(Xb)
            y_true.extend(yb.tolist())
            y_pred.extend(out.argmax(dim=1).tolist())
            probs.extend(F.softmax(out, dim=1).tolist())
            attns.extend(att.numpy())

    print("Test Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(f"{SEQ_RESULT_DIR}/confusion_matrix.png")
    plt.close()

    # ROC
    y_bin = label_binarize(y_true, classes=[0,1,2])
    probs = np.array(probs)
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{SEQ_RESULT_DIR}/roc_curve.png")
    plt.close()

    # Attention Heatmap
    mean_attn = np.mean(attns, axis=0)
    sns.heatmap(mean_attn[np.newaxis, :], cmap='viridis', cbar=True)
    plt.title("Mean Attention")
    plt.savefig(f"{SEQ_RESULT_DIR}/attention_weights.png")
    plt.close()

    # PCA Visualization
    ctx_vecs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 64):
            out, _ = model.lstm(X[i:i+64])
            ctx, _ = model.attn(out)
            ctx_vecs.append(ctx)
    ctx_vecs = torch.cat(ctx_vecs, dim=0).numpy()
    pca_plot(ctx_vecs)
    tsne_plot(ctx_vecs)
    umap_plot(ctx_vecs, 100)