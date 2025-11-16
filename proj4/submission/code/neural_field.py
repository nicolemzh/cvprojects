import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os

class PositionalEncoding:
    def __init__(self, L=10):
        self.L = L
        self.output_dim = 4 * L + 2

    def encode(self, x):
        encoding = [x]
        for i in range(self.L):
            freq = 2 ** i
            encoding.append(torch.sin(freq * x)) # torch.sin(freq * np.pi * x))
            encoding.append(torch.cos(freq * x)) # torch.cos(freq * np.pi * x))
        return torch.cat(encoding, dim=-1)
    
class NeuralField(nn.Module):
    def __init__(self, L=10, hidden_dim=256):
        super().__init__()

        self.pe = PositionalEncoding(L)

        self.layers = nn.Sequential(
            nn.Linear(self.pe.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3), # output 3 rgb channels
            nn.Sigmoid()
        )

    def forward(self, x):
        encoding = self.pe.encode(x)
        return self.layers(encoding)
    
class DataLoader:
    def __init__(self, file, batch_size):
        img = Image.open(file).convert('RGB')
        self.h, self.w = img.size[1], img.size[0]
        self.img_arr = np.array(img) / 255.0 # normalize

        y_coords, x_coords = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing='ij') # np.mgrid[0:self.h, 0:self.w]
        self.coords = np.stack([
            x_coords.flatten() / self.w,
            y_coords.flatten() / self.h
        ], axis=-1).astype(np.float32)

        self.rgbs = self.img_arr.reshape(-1, 3)
        self.num_pixels = self.h * self.w
        self.batch_size = batch_size
    
    def sample(self):
        idx = np.random.choice(self.num_pixels, self.batch_size, replace=False)

        coords = torch.from_numpy(self.coords[idx].astype(np.float32))
        rgbs = torch.from_numpy(self.rgbs[idx].astype(np.float32))
        
        return coords, rgbs
    
    def all(self):
        coords = torch.from_numpy(self.coords.astype(np.float32))
        rgbs = torch.from_numpy(self.rgbs.astype(np.float32))
        
        return coords, rgbs
    
def reconstruct_image(model, dataloader, device, path):
    model.eval()

    with torch.no_grad():
        coords, _ = dataloader.all() # gets all coords
        coords = coords.to(device)
        rgbs = model(coords).cpu().numpy()

    # reshape
    pred_img = rgbs.reshape(dataloader.h, dataloader.w, 3)
    pred_img = (pred_img * 255).astype(np.uint8)
    
    Image.fromarray(pred_img).save(path)
    
    model.train()

def visualize_progression(file, psnrs, L, width):
    original_img = Image.open(f"{file}.jpg")
    
    iterations = sorted(psnrs.keys())
    num_images = len(iterations) + 1 

    fig, axes = plt.subplots(1, num_images, figsize=(2.5 * num_images, 3))
    fig.suptitle(f"{file}, L={L}, width={width}", fontsize=12)

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=9, fontweight='bold')
    axes[0].axis('off')
    
    for idx, iter_num in enumerate(iterations, start=1):
        img_path = f"output/{file}/L{L}W{width}/iter_{iter_num}.png"
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(
                f"Iteration {iter_num}\nPSNR: {psnrs[iter_num]:.2f} dB",
                fontsize=9
            )
        axes[idx].axis('off')
    
    # save figure
    plt.tight_layout()
    output_path = f"output/{file}/L{L}W{width}/progression.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
def psnr(mse):
    return 10 * np.log10(1.0 / mse)
    
'''
 Now that you have the network (MLP) and the dataloader, you need to define the loss 
 function and the optimizer before you can start training your network. You will use 
 mean squared error loss (MSE) (torch.nn.MSELoss) between the predicted color and the 
 groundtruth color. Train your network using Adam (torch.optim.Adam) with a learning 
 rate of 1e-2. Run the training loop for 1000 to 3000 iterations with a batch size of 
 10k. For the metric, MSE is a good one but it is more common to use Peak signal-to-noise 
 ratio (PSNR) when it comes to measuring the reconstruction quality of a image.
 '''
def train(file, L=10, hidden_dim=512, lr=1e-2, batch_size=10000, iters=3000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloader = DataLoader(f"{file}.jpg", batch_size)

    model = NeuralField(L, hidden_dim).to(device)
    print(f"L={L}, Hidden dimension: {hidden_dim}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    psnr_records = []
    mse_records = []
    saved_psnrs = {}

    print(f"Training for {iters} iterations")
    for iter in tqdm(range(iters)):
        batch_coords, batch_rgbs = dataloader.sample()
        batch_coords = batch_coords.to(device)
        batch_rgbs = batch_rgbs.to(device)

        # forward pass
        pred_rgbs = model(batch_coords)
        loss = criterion(pred_rgbs, batch_rgbs)

        # backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse = loss.item()
        cur_psnr = psnr(mse)

        mse_records.append(mse)
        psnr_records.append(cur_psnr)
        
        if iter == 0 or (iter + 1) % 100 == 0:
            tqdm.write(f"Iter {iter+1}/{iters}; Loss: {mse}; PSNR: {cur_psnr} dB")
        if iter == 0 or (iter + 1) % 250 == 0:
            reconstruct_image(model, dataloader, device, f"output/{file}/L{L}W{hidden_dim}/iter_{iter+1}.png")
            saved_psnrs[iter+1] = cur_psnr

    torch.save(model.state_dict(), os.path.join(f"output/{file}/L{L}W{hidden_dim}/model.pth"))

    plt.figure(figsize=(10, 5))
    plt.plot(psnr_records)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Training Progression')
    plt.savefig(f"output/{file}/L{L}W{hidden_dim}/psnr_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    visualize_progression(file, saved_psnrs, L, hidden_dim)
    
    print(f"\nFinal PSNR: {psnr_records[-1]}")
    
    return model, psnr_records, mse_records

if __name__ == "__main__":
    image = "bird"
    model, psnr_records, mse_records = train(
        image,
        L=20,
        hidden_dim=128,
        lr=1e-2,
        batch_size=10000,
        iters=3000
    )

    