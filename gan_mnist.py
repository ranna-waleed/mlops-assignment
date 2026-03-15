import pandas as pd
import torch
import torch.nn as nn
import matplotlib
import mlflow
import mlflow.pytorch
matplotlib.use("Agg")
import matplotlib.pyplot as plt

latent_dim = 100
image_dim = 784
batch_size = 128
epochs = 20
lr = 0.0002

device = "cuda" if torch.cuda.is_available() else "cpu"

data = pd.read_csv("fashion_mnist_pixels.csv").values
data = torch.tensor(data, dtype=torch.float32).to(device)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=lr)
opt_D = torch.optim.Adam(D.parameters(), lr=lr)

# MLflow experiment
mlflow.set_experiment("Assignment3_Rana")

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("latent_dim", latent_dim)

    mlflow.set_tag("student_id", "202201737")

    for epoch in range(epochs):

        perm = torch.randperm(len(data))
        correct = 0
        total = 0

        for i in range(0, len(data), batch_size):

            real = data[perm[i:i+batch_size]]
            bsz = real.size(0)

            real_labels = torch.ones(bsz, 1).to(device)
            fake_labels = torch.zeros(bsz, 1).to(device)

            # Train Discriminator
            z = torch.randn(bsz, latent_dim).to(device)
            fake = G(z)

            real_pred = D(real)
            fake_pred = D(fake.detach())

            loss_D = criterion(real_pred, real_labels) + criterion(fake_pred, fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Accuracy
            correct += (real_pred > 0.5).sum().item()
            correct += (fake_pred < 0.5).sum().item()
            total += 2 * bsz

            # Train Generator
            loss_G = criterion(D(fake), real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {loss_D.item():.4f}  G Loss: {loss_G.item():.4f}  D Acc: {acc:.2f}%")

        # Log metrics every epoch
        mlflow.log_metric("D_loss", loss_D.item(), step=epoch)
        mlflow.log_metric("G_loss", loss_G.item(), step=epoch)
        mlflow.log_metric("D_accuracy", acc, step=epoch)

    print("\nFinal Results:")
    print(f"Generator Loss: {loss_G.item():.4f}")
    print(f"Discriminator Loss: {loss_D.item():.4f}")
    print(f"Discriminator Accuracy: {acc:.2f}%")

    # Generate images
    G.eval()
    z = torch.randn(16, latent_dim).to(device)
    generated = G(z).detach().cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(6,6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i], cmap="gray")
        ax.axis("off")

    plt.savefig("generated.png")
    print("Image saved as generated.png")

    # Log artifact
    mlflow.log_artifact("generated.png")

    # Log model
    mlflow.pytorch.log_model(G, "generator_model")
