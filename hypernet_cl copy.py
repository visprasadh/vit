import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bar
import math

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from timm import create_model

from vit import VisionTransformer
from lora_vit import LoRA

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

gpu = 0
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Load pretrained model from timm
pretrained_model = create_model('vit_base_patch16_224', pretrained=True)
print("Loaded pretrained model from timm")

n_transformer_layers = 12

# Initialize your custom VisionTransformer
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    num_classes=10,  # CIFAR10 has 10 classes
    dim=768,  # Same as vit_base
    depth=n_transformer_layers,
    heads=12,
    mlp_dim=3072,
    dropout=0.1
)

# Transfer weights from pretrained model to your custom model
def transfer_weights(pretrained_model, custom_model):
    print("Transferring weights from pretrained model to custom model...")
    
    # Map for patch embedding
    custom_model.patch_embedding.weight.data = pretrained_model.patch_embed.proj.weight.data
    custom_model.patch_embedding.bias.data = pretrained_model.patch_embed.proj.bias.data
    
    # Map for position embedding (accounting for extra CLS token)
    custom_model.position_embedding.data = pretrained_model.pos_embed.data
    
    # Map for class token
    custom_model.cls_token.data = pretrained_model.cls_token.data
    
    # Map transformer blocks
    for i in range(12):  # Assuming 12 blocks in both models
        # Layer Norm 1
        custom_model.transformer_blocks[i][0].weight.data = pretrained_model.blocks[i].norm1.weight.data
        custom_model.transformer_blocks[i][0].bias.data = pretrained_model.blocks[i].norm1.bias.data
        
        # Attention
        # QKV projection
        custom_model.transformer_blocks[i][1].qkv.weight.data = pretrained_model.blocks[i].attn.qkv.weight.data
        custom_model.transformer_blocks[i][1].qkv.bias.data = pretrained_model.blocks[i].attn.qkv.bias.data
        
        # Projection after attention
        custom_model.transformer_blocks[i][1].proj.weight.data = pretrained_model.blocks[i].attn.proj.weight.data
        custom_model.transformer_blocks[i][1].proj.bias.data = pretrained_model.blocks[i].attn.proj.bias.data
        
        # Layer Norm 2
        custom_model.transformer_blocks[i][2].weight.data = pretrained_model.blocks[i].norm2.weight.data
        custom_model.transformer_blocks[i][2].bias.data = pretrained_model.blocks[i].norm2.bias.data
        
        # MLP layers in Feed Forward
        custom_model.transformer_blocks[i][3].net[0].weight.data = pretrained_model.blocks[i].mlp.fc1.weight.data
        custom_model.transformer_blocks[i][3].net[0].bias.data = pretrained_model.blocks[i].mlp.fc1.bias.data
        custom_model.transformer_blocks[i][3].net[3].weight.data = pretrained_model.blocks[i].mlp.fc2.weight.data
        custom_model.transformer_blocks[i][3].net[3].bias.data = pretrained_model.blocks[i].mlp.fc2.bias.data
    
    # Final norm layer
    custom_model.mlp_head[0].weight.data = pretrained_model.norm.weight.data
    custom_model.mlp_head[0].bias.data = pretrained_model.norm.bias.data
    
    # Don't copy the final classification head as we're fine-tuning for CIFAR-10 (10 classes)
    # Instead, initialize it randomly (which is already done by PyTorch)
    
    print("Weight transfer complete!")
    return custom_model

# Transfer weights
model = transfer_weights(pretrained_model, model)

# for name, module in model.named_modules():
#     print(f"Module Name: {name}, Module: {module}")

model.to(device)

criterion = nn.CrossEntropyLoss()

# Proper LoRA initialization with standard dimensions
r = 8  # Low-rank dimension (standard LoRA parameter)
lora_alpha = 16  # Scaling factor for LoRA (matching the model's default)

# Initialize LoRA parameters with proper dimensions
lora_A = torch.zeros(n_transformer_layers, r, 768).to(device)  # r x in_features
lora_A.requires_grad = True
lora_B = torch.zeros(n_transformer_layers, 2304, r).to(device)  # out_features x r
lora_B.requires_grad = True

# Use proper initialization (similar to standard LoRA)
for i in range(n_transformer_layers):
    nn.init.kaiming_uniform_(lora_A[i], a=math.sqrt(5))
    # B is initialized to zeros as is standard in LoRA

model.freeze_parameters()

# Create a list of parameters to optimize, including lora_A and lora_B
optimizer = optim.Adam([lora_A, lora_B], lr=0.001)

print("Training the model...")
print(f"LoRA dimensions - A: {lora_A.shape}, B: {lora_B.shape}, rank: {r}, alpha: {lora_alpha}")

def train_one_epoch(model, lora_A, lora_B, train_loader, criterion, optimizer, device, epoch):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, lora_A, lora_B)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Print every 10 batches to avoid excessive output
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
            print(f"Lora A grad norm: {lora_A.grad.norm().item() if lora_A.grad is not None else 'None'}")
            print(f"Lora B grad norm: {lora_B.grad.norm().item() if lora_B.grad is not None else 'None'}")

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

def test_model(model, lora_A, lora_B, test_loader, device):
    model.eval()  # Set model to evaluation mode
    total = 0
    correct = 0
    
    with torch.no_grad():  # Disable gradient calculation
        # Use tqdm to show progress for testing
        for i, (images, labels) in enumerate(tqdm(test_loader, desc='Testing')):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, lora_A, lora_B)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total, correct

epochs = 10

for epoch in range(epochs):
    train_one_epoch(model, lora_A, lora_B, train_loader, criterion, optimizer, device, epoch)
    total, correct = test_model(model, lora_A, lora_B, test_loader, device)
    print(f'Epoch {epoch+1} - Accuracy: {100 * correct / total:.2f}%')

print(f'Final accuracy of the model on the test images: {100 * correct / total:.2f}%')
