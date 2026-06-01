import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.encoder(x))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, latent_dim, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, embed_dim: int = 128):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViTClassifier(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, embed_dim: int = 128, depth: int = 4, num_heads: int = 4, mlp_dim: int = 256, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.head(self.norm(x[:, 0]))


def build_model(name: str, cfg: dict, num_classes: int = 10):
    name = name.lower()
    if name == "cnn":
        return CNNClassifier(num_classes=num_classes)
    if name == "autoencoder":
        return ConvAutoencoder(latent_dim=cfg.get("latent_dim", 128))
    if name == "vit":
        return ViTClassifier(num_classes=num_classes, **cfg)
    raise ValueError(f"Unknown model: {name}")
