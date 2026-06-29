import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import Decoder

parser = argparse.ArgumentParser(description="Generate gambar dari latent vector menggunakan Decoder")
parser.add_argument("--decoder", required=True, help="Path ke file decoder .pth")
parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension decoder")
parser.add_argument("--z1", type=float, help="Nilai latent ke-1 (untuk dim=2)")
parser.add_argument("--z2", type=float, help="Nilai latent ke-2 (untuk dim=2)")
parser.add_argument("--latent", type=str, help="Daftar nilai latent dipisah koma, e.g. '0.5,-1.2,0.3,0.8'")
parser.add_argument("--outdir", default=".", help="Directory untuk menyimpan output")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

decoder = Decoder(args.latent_dim).to(device)
decoder.load_state_dict(torch.load(args.decoder, map_location=device, weights_only=True))
decoder.eval()

if args.latent:
    values = [float(x) for x in args.latent.split(",")]
    if len(values) != args.latent_dim:
        print(f"Error: latent_dim={args.latent_dim} tapi diberi {len(values)} nilai")
        exit(1)
    latent = torch.tensor([values], device=device)
elif args.z1 is not None and args.z2 is not None:
    if args.latent_dim != 2:
        print(f"Error: --z1/--z2 hanya untuk latent_dim=2 (dimiliki {args.latent_dim})")
        exit(1)
    latent = torch.tensor([[args.z1, args.z2]], device=device)
else:
    print("Error: beri --z1 --z2 (untuk dim=2) atau --latent (untuk dim berapapun)")
    exit(1)

with torch.no_grad():
    output = decoder(latent)
    generated = output.squeeze().cpu()

plt.imshow(generated, cmap="gray")
plt.title(f"Generated from latent {latent.cpu().numpy().tolist()[0]}")
plt.axis("off")
plt.savefig(f"{args.outdir}/generated_image.png", bbox_inches="tight")
print(f"Output saved: {args.outdir}/generated_image.png")
