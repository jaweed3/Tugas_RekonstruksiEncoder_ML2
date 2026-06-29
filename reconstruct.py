import argparse
import torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import Autoencoder

parser = argparse.ArgumentParser(description="Rekonstruksi Fashion-MNIST dengan Autoencoder")
parser.add_argument("--model", required=True, help="Path ke file model .pth")
parser.add_argument("--index", type=int, default=0, help="Index data Fashion-MNIST (0-9999)")
parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension model")
parser.add_argument("--outdir", default=".", help="Directory untuk menyimpan output")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder(args.latent_dim).to(device)
model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
img, _ = dataset[args.index]

with torch.no_grad():
    input_tensor = img.unsqueeze(0).to(device)
    output_tensor = model(input_tensor)
    reconstructed = output_tensor.squeeze().cpu()

original = img.squeeze()

plt.imsave(f"{args.outdir}/original.png", original, cmap="gray")
plt.imsave(f"{args.outdir}/reconstructed.png", reconstructed, cmap="gray")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(original, cmap="gray")
axes[0].set_title(f"Original (index {args.index})")
axes[0].axis("off")

axes[1].imshow(reconstructed, cmap="gray")
axes[1].set_title("Reconstructed")
axes[1].axis("off")

axes[2].imshow(original, cmap="gray")
axes[2].imshow(reconstructed, cmap="gray", alpha=0.5)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(f"{args.outdir}/comparison.png", bbox_inches="tight")
print(f"Output saved: {args.outdir}/original.png, reconstructed.png, comparison.png")
