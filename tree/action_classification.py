from read_data import get_train_data,get_test_data
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

data_path = './HAR'
train_path = 'train'
test_path = 'test'

xtrain,ytrain = get_train_data(data_path,train_path)
xtest,ytest =get_test_data(data_path,test_path)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, num_classes=6):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        # x shape: (batch, 128, 9)

        out, (h_n, c_n) = self.lstm(x)
        last_output = out[:, -1, :]   # (batch, hidden_size)

        logits = self.fc(last_output) # (batch, 6)

        return logits

def train_model(model, train_loader, test_loader, epochs=30, lr=1e-3):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0

    for epoch in range(epochs):

        # ======== TRAIN ========
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ======== EVALUATE ========
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)

                outputs = model(xb)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {total_loss/len(train_loader):.4f} "
              f"Macro-F1: {macro_f1:.4f}")

        # 保存最佳模型
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), "best_action_model.pth")

    print("Best Macro-F1:", best_f1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 轉 tensor
xtrain = torch.tensor(xtrain, dtype=torch.float32)
ytrain = torch.tensor(ytrain, dtype=torch.long).squeeze()

xtest = torch.tensor(xtest, dtype=torch.float32)
ytest = torch.tensor(ytest, dtype=torch.long).squeeze()

train_dataset = TensorDataset(xtrain, ytrain)
test_dataset = TensorDataset(xtest, ytest)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

model = LSTMClassifier()
train_model(model, train_loader, test_loader, epochs=30)

model.load_state_dict(torch.load("best_action_model.pth"))
model.eval()
model.cpu()
dummy_input = torch.randn(1, 128, 9)
torch.onnx.export(
    model,
    dummy_input,
    "lstm_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=17
)



def save_cpp_test_samples(xtest, ytest, save_dir="cpp_test_samples"):
    os.makedirs(save_dir, exist_ok=True)

    # 隨機選 2 個 index
    # random choose 2 index
    indices = np.random.choice(len(xtest), 2, replace=False)

    for i, idx in enumerate(indices):
        sample = xtest[idx]      # (128,9)
        label = ytest[idx]

        print(f"Saving sample {i}, label = {label}")

        # ===== 保存為 keep as  txt =====
        txt_path = os.path.join(save_dir, f"sample_{i}.txt")
        np.savetxt(txt_path, sample.reshape(-1), fmt="%.6f")

        # ===== 保存為 keep as  bin（float32）=====
        bin_path = os.path.join(save_dir, f"sample_{i}.bin")
        sample.astype(np.float32).tofile(bin_path)

        # ===== 保存 label =====
        with open(os.path.join(save_dir, f"label_{i}.txt"), "w") as f:
            f.write(str(int(label)))

    print("Done.")

save_cpp_test_samples(xtest, ytest, save_dir="cpp_test_samples")


