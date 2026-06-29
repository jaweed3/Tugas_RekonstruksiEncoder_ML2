# Tugas Autoencoder Fashion-MNIST

**Nama:** [Nama]
**NIM:** [NIM]

## Versi Python dan Library

- Python 3.14.5
- PyTorch 2.12.0
- torchvision 0.27.0
- matplotlib 3.11.0
- numpy 2.4.6

## Cara Menjalankan Training di Kaggle

1. Buka [Kaggle](https://www.kaggle.com/) dan buat Notebook baru.
2. Upload file `autoencoder-fashion-mnist.ipynb` atau copy kode ke dalam notebook.
3. Set accelerator ke GPU (Settings > Accelerator > GPU P100).
4. Jalankan semua cell secara berurutan.
5. Model akan dilatih untuk 3 latent dimension: 2, 8, dan 32.
6. Setelah selesai, download file `.pth` dan hasil gambar.

## Cara Menjalankan Rekonstruksi dari Terminal

### Opsi A: Rekonstruksi dengan Autoencoder Penuh

```bash
python reconstruct.py --model models/autoencoder_fashion_mnist_dim32.pth --latent_dim 32 --index 25 --outdir results
```

Output: `original.png`, `reconstructed.png`, `comparison.png`

### Opsi B: Generasi Citra dengan Decoder (latent dim = 2)

```bash
python generate_from_decoder.py --decoder models/decoder_fashion_mnist_dim2.pth --latent_dim 2 --z1 0.5 --z2 -1.2 --outdir results
```

### Opsi B: Generasi Citra dengan Decoder (latent dim = 8)

```bash
python generate_from_decoder.py --decoder models/decoder_fashion_mnist_dim8.pth --latent_dim 8 --latent "0.5,-1.2,0.3,0.8,0.1,-0.5,0.7,1.0" --outdir results
```

Output: `generated_image.png`

## Daftar File Output

| File | Deskripsi |
|------|-----------|
| `autoencoder-fashion-mnist.ipynb` | Notebook training untuk Kaggle |
| `models/autoencoder_fashion_mnist_dim*.pth` | Model autoencoder lengkap |
| `models/encoder_fashion_mnist_dim*.pth` | Encoder saja |
| `models/decoder_fashion_mnist_dim*.pth` | Decoder saja |
| `reconstruct.py` | Program rekonstruksi dari terminal |
| `generate_from_decoder.py` | Program generasi dari decoder |
| `results/training_loss_comparison.png` | Grafik loss training |
| `results/reconstruction_comparison.png` | Perbandingan rekonstruksi |
| `results/original.png` | Gambar asli |
| `results/reconstructed.png` | Gambar rekonstruksi |
| `results/comparison.png` | Side-by-side original vs rekonstruksi |
| `results/generated_image.png` | Gambar hasil generasi dari decoder |
| `laporan.pdf` | Laporan tugas |
