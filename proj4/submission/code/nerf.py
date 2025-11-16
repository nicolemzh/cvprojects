import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import math
import imageio
import os
import glob
import time

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
    
    def encode(self, x):
        encoded = [x]
        for l in range(self.L):
            encoded.append(torch.sin(2**l * np.pi * x))
            encoded.append(torch.cos(2**l * np.pi * x))
        return torch.cat(encoded, dim=-1)
    
def transform(c2w, x_c):
    ones = np.ones_like(x_c[..., :1])
    x_c_h = np.concatenate([x_c, ones], axis=-1)
    if c2w.ndim == 2:
        x_w_h = (c2w @ x_c_h.T).T
    else:
        x_w_h = np.matmul(c2w, x_c_h[..., None]).squeeze(-1)
    x_w = x_w_h[..., :3] / x_w_h[..., 3:]
    return x_w

def pixel_to_camera(K, uv, s):
    ones = np.ones_like(uv[..., :1])
    uv_h = np.concatenate([uv, ones], axis=-1)
    K_inv = np.linalg.inv(K)
    x_c_unit = (K_inv @ uv_h.T).T
    x_c = x_c_unit * s
    return x_c

def pixel_to_ray(K, c2w, uv):
    ray_o = c2w[:3, 3]
    if uv.ndim == 2:
        ray_o = np.tile(ray_o, (uv.shape[0], 1))
    elif uv.ndim == 3:
        ray_o = np.tile(ray_o.reshape(1, 1, 3), (uv.shape[0], uv.shape[1], 1))
    x_c = pixel_to_camera(K, uv, s=1.0)
    x_w = transform(c2w, x_c)
    ray_d = x_w - ray_o
    ray_d = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True)
    return ray_o, ray_d

def sample_along_rays(rays_o, rays_d, near=2.0, far=6.0, n_samples=64, perturb=True):
    N = rays_o.shape[0]
    t = np.linspace(near, far, n_samples)
    t = np.tile(t, (N, 1))
    
    if perturb:
        t_width = (far - near) / n_samples
        t = t + np.random.rand(*t.shape) * t_width
    
    x = rays_o[:, None, :] + rays_d[:, None, :] * t[:, :, None]
    return x

class NeRF(nn.Module):
    def __init__(self, L_x=10, L_d=4, hidden_dim=256):
        super().__init__()

        self.L_x = L_x
        self.L_d = L_d
        self.hidden_dim = hidden_dim

        self.pe_x = PositionalEncoding(L_x)
        self.pe_d = PositionalEncoding(L_d)

        x_dim = 3 * (2 * L_x) + 3 
        d_dim = 3 * (2 * L_d) + 3

        self.layers1 = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.layers2 = nn.Sequential(
            nn.Linear(hidden_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.density_layers = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )

        self.rgb_layers = nn.Sequential(
            nn.Linear(hidden_dim + d_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x, d):
        x_encoding = self.pe_x.encode(x)
        d_encoding = self.pe_d.encode(d)

        h = self.layers1(x_encoding)

        # skip connection
        h = torch.cat([h, x_encoding], dim=-1)
        h = self.layers2(h)

        density = self.density_layers(h)

        rgb = torch.cat([h, d_encoding], dim=-1)
        rgb = self.rgb_layers(rgb)

        return rgb, density

def volrend(sigmas, rgbs, step_size):
    alphas = 1.0 - torch.exp(-sigmas * step_size) # 1 - exp(-sigma * delta)
    sigma_delta = sigmas * step_size # sigma_j * delta_j
    
    cumsum_sigma_delta = torch.cumsum(sigma_delta, dim=1) 

    # shift from 0 to i - 1
    cumsum_shifted = torch.cat([torch.zeros_like(cumsum_sigma_delta[:, :1, :]), cumsum_sigma_delta[:, :-1, :]], dim=1)
    T = torch.exp(-cumsum_shifted)

    weights = T * alphas # T_i * alpha_i
    rendered_colors = torch.sum(weights * rgbs, dim=1)
    
    return rendered_colors

def psnr(mse):
    return 10 * np.log10(1.0 / mse)

def mse_to_psnr(mse):
    return -10.0 * math.log10(max(mse, 1e-12))

def reconstruct_image(model, H, W, K, c2w, near=2.0, far=6.0, num_ray_samples=64, device='cuda', chunk_size=512):
    model.eval()
    
    # generate all rays
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
    uv = torch.stack([x + 0.5, y + 0.5], dim=-1).reshape(-1, 2)
    
    K_np = K if isinstance(K, np.ndarray) else K.cpu().numpy()
    c2w_np = c2w if isinstance(c2w, np.ndarray) else c2w.cpu().numpy()
    
    rays_o, rays_d = pixel_to_ray(K_np, c2w_np, uv.numpy())
    rays_o = torch.from_numpy(rays_o).float().to(device)
    rays_d = torch.from_numpy(rays_d).float().to(device)
    
    all_rgb = []
    
    with torch.no_grad():
        for i in range(0, len(rays_o), chunk_size):
            rays_o_chunk = rays_o[i:i+chunk_size]
            rays_d_chunk = rays_d[i:i+chunk_size]
            
            # sample points
            x_np = sample_along_rays(rays_o_chunk.cpu().numpy(), rays_d_chunk.cpu().numpy(), 
                                       near=near, far=far, n_samples=num_ray_samples, perturb=False)
            x = torch.from_numpy(x_np).float().to(device)
            dirs = rays_d_chunk[:, None, :].expand(x.shape)
            
            # forward pass
            rgb, sigma = model(x, dirs)
            step_size = (far - near) / num_ray_samples
            rendered_rgb = volrend(sigma, rgb, step_size)
            
            all_rgb.append(rendered_rgb.cpu())
    
    all_rgb = torch.cat(all_rgb, dim=0)
    model.train()
    
    return all_rgb.reshape(H, W, 3).numpy()

def train_nerf(
    imgs_train,
    c2ws_train,
    imgs_val,
    c2ws_val,
    K,
    num_iters=1000,
    batch_size=10000,
    lr=5e-4,
    num_ray_samples=64,
    near=2.0,
    far=6.0,
    chunk_size=8192,
    file=""):
    
    H, W = imgs_train.shape[1:3]
    print(f"Using {device}")

    model = NeRF(L_x=10, L_d=4, hidden_dim=256).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    imgs_train_t = torch.from_numpy(imgs_train).float().to(device)
    c2ws_train_t = torch.from_numpy(c2ws_train).float()
    K_np = K if isinstance(K, np.ndarray) else K.cpu().numpy()

    psnr_records = []
    psnr_val_records = []
    mse_records = []
    it_hist = []
    val_psnr_hist = []
    render_snapshots = []

    start_time = time.time()

    for iter in range(1, num_iters + 1):
        model.train()
        
        img_idxs = np.random.randint(0, len(imgs_train), batch_size)
        u_idx = np.random.randint(0, W, batch_size)
        v_idx = np.random.randint(0, H, batch_size)
        
        real_rgb = imgs_train_t[img_idxs, v_idx, u_idx]
        
        rays_o_list = []
        rays_d_list = []
        for img_idx, u, v in zip(img_idxs, u_idx, v_idx):
            uv = np.array([[u + 0.5, v + 0.5]])
            c2w_np = c2ws_train_t[img_idx].numpy()
            ray_o, ray_d = pixel_to_ray(K_np, c2w_np, uv)
            rays_o_list.append(ray_o)
            rays_d_list.append(ray_d)
        
        rays_o = np.concatenate(rays_o_list, axis=0)
        rays_d = np.concatenate(rays_d_list, axis=0)
        
        rays_o = torch.from_numpy(rays_o).float().to(device)
        rays_d = torch.from_numpy(rays_d).float().to(device)

        x = sample_along_rays(
            rays_o.cpu().numpy(),
            rays_d.cpu().numpy(),
            near=near,
            far=far,
            n_samples=num_ray_samples,
            perturb=True
        )

        x = torch.from_numpy(x).float().to(device)
        dirs = rays_d[:, None, :].expand(x.shape)
        
        # forward pass
        rgb, sigma = model(x, dirs)
        step_size = (far - near) / num_ray_samples
        pred_rgb = volrend(sigma, rgb, step_size)
        
        # backward pass
        loss = criterion(pred_rgb, real_rgb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mse = loss.item()
        cur_psnr = psnr(mse)
        psnr_records.append(cur_psnr)
        mse_records.append(mse)

        if iter % 200 == 0 or iter == 1:
            val_psnrs = []
            model.eval()
            with torch.no_grad():
                for i in range(min(6, len(imgs_val))):
                    img_recon = reconstruct_image(model, H, W, K_np, c2ws_val[i],
                                                  near=near, far=far, 
                                                  num_ray_samples=num_ray_samples,
                                                  device=device, chunk_size=chunk_size)
                    mse_val = np.mean((img_recon - imgs_val[i]) ** 2)
                    val_psnrs.append(psnr(mse_val))
            
            psnr_avg_val = np.mean(val_psnrs)
            it_hist.append(iter)
            val_psnr_hist.append(psnr_avg_val)
            psnr_val_records.append((iter, psnr_avg_val))
            
            elapsed = time.time() - start_time
            print(f"iter {iter:5d}; train PSNR {cur_psnr:6.2f} dB; val PSNR {psnr_avg_val:6.2f} dB")
            model.train()

        if iter % 200 == 0 or iter == num_iters:
            model.eval()
            with torch.no_grad():
                for i in range(min(6, len(imgs_val))):
                    img_recon = reconstruct_image(model, H, W, K_np, c2ws_val[i],
                                                  near=near, far=far, 
                                                  num_ray_samples=num_ray_samples,
                                                  device=device, chunk_size=chunk_size)
                    
                    img = (img_recon * 255).astype(np.uint8)
                    Image.fromarray(img).save(f"nerf_output/{file}/val_{i}_iter_{iter:04d}.png")
                
                img_recon_0 = reconstruct_image(model, H, W, K_np, c2ws_val[0],
                                                near=near, far=far, 
                                                num_ray_samples=num_ray_samples,
                                                device=device, chunk_size=chunk_size)
                render_snapshots.append((iter, torch.from_numpy(img_recon_0)))

            torch.save({
                'iteration': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_psnr': psnr(mse),
                'val_psnr': psnr_avg_val,
            }, f"nerf_output/{file}/checkpoint_iter_{iter:04d}.pth")
            model.train()
    
    # plot training curves
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(mse_records)
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Progression')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(psnr_records)
    plt.xlabel('Iteration')
    plt.ylabel('Training PSNR (dB)')
    plt.title('PSNR Training Progression')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if psnr_val_records:
        iters, psnrs = zip(*psnr_val_records)
        plt.plot(iters, psnrs, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Validation PSNR (dB)')
        plt.title('PSNR Validation Progression')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"nerf_output/{file}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    visualize_progression(file)
    
    return model, (it_hist, val_psnr_hist, render_snapshots)

def visualize_progression(file):
    img_paths = sorted(glob.glob(f'nerf_output/{file}/val_0_iter_*.png'))
    
    valid_iters = []
    for path in img_paths:
        iter_num = int(path.split('_iter_')[-1].replace('.png', ''))
        valid_iters.append((iter_num, path))
    
    valid_iters.sort()
    
    num_images = min(6, len(valid_iters))
    indices = np.linspace(0, len(valid_iters)-1, num_images, dtype=int)
    selected = [valid_iters[i] for i in indices]
    
    images = []
    iter_labels = []
    for iter_num, path in selected:
        img = Image.open(path)
        images.append(img)
        iter_labels.append(iter_num)

    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(3*num_images, 3))
    
    if num_images == 1:
        axes = [axes]
    
    for ax, img, iter_num in zip(axes, images, iter_labels):
        ax.imshow(img)
        ax.set_title(f'Iteration {iter_num}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f"{file} Training Progression", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = f'nerf_output/{file}/training_progression.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def plot_results(it_hist, val_psnr_hist, render_snapshots, gt_image=None):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(it_hist, val_psnr_hist)
    plt.title("Validation PSNR")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")

    if gt_image is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(gt_image)
        plt.axis('off')

    if render_snapshots:
        plt.subplot(1, 3, 3)
        img_np = render_snapshots[-1][1].clamp(0, 1).cpu().numpy() if torch.is_tensor(render_snapshots[-1][1]) else render_snapshots[-1][1]
        plt.imshow(img_np)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    num_snapshots = len(render_snapshots)
    fig, axes = plt.subplots(1, num_snapshots, figsize=(3 * num_snapshots, 3))
    if num_snapshots == 1: axes = [axes]
    for i, (it, img_pred) in enumerate(render_snapshots):
        img_np = img_pred.clamp(0, 1).cpu().numpy() if torch.is_tensor(img_pred) else img_pred
        axes[i].imshow(img_np)
        axes[i].set_title(f"Iteration: {it}")
        axes[i].axis('off')
    plt.show()

def create_gif(num_views=6, file='lego', duration=300):
    all_paths = sorted(glob.glob(f'nerf_output/{file}/val_*_iter_*.png'))
    
    iterations = set()
    for path in all_paths:
        iter_num = int(path.split('_iter_')[-1].replace('.png', ''))
        iterations.add(iter_num)
    
    last_iter = max(iterations)
    
    images = []
    for i in range(num_views):
        path = f'nerf_output/{file}/val_{i}_iter_{last_iter:04d}.png'
        if os.path.exists(path):
            img = imageio.imread(path)
            images.append(img)
    
    output_path = f'nerf_output/{file}/{file}.gif'
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    
    return output_path

# CODE PROVIDED
def look_at_origin(pos):
    forward = -pos / np.linalg.norm(pos)
    up_tmp = np.array([0, 1, 0])
    right = np.cross(up_tmp, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w

def rot_y(phi_rad):
    return np.array([
        [math.cos(phi_rad), 0, math.sin(phi_rad), 0],
        [0, 1, 0, 0],
        [-math.sin(phi_rad), 0, math.cos(phi_rad), 0],
        [0, 0, 0, 1],
    ])

def render_orbit_video(model, K, H, W, near, far, n_samples, c2ws_test, filename, chunk=8192):
    frames = []
    K_np = K if isinstance(K, np.ndarray) else K.cpu().numpy()
    
    for i, c2w_np in enumerate(c2ws_test):
        img = reconstruct_image(model, H, W, K_np, c2w_np, near=near, far=far,
                               num_ray_samples=n_samples, device=device, chunk_size=chunk)
        img_np = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(img_np)

    imageio.mimsave(filename, frames, fps=30)

def render_novel_orbit(model, K, H, W, near, far, n_samples, start_pos, filename, num_frames=60, chunk=8192):
    frames = []
    base_c2w = look_at_origin(start_pos)
    K_np = K if isinstance(K, np.ndarray) else K.cpu().numpy()

    for phi_deg in np.linspace(0., 360., num_frames, endpoint=False):
        phi_rad = phi_deg / 180. * np.pi
        extrinsic = rot_y(phi_rad) @ base_c2w

        img = reconstruct_image(model, H, W, K_np, extrinsic, near=near, far=far,
                               num_ray_samples=n_samples, device=device, chunk_size=chunk)
        img_np = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(img_np)

    imageio.mimsave(filename, frames, fps=60)

def main():
    data_file = "nerf/cube_data.npz"
    file_name = "cube"
    
    data = np.load(data_file)

    images_train = data["images_train"] / 255.0
    c2ws_train = data["c2ws_train"]
    images_val = data["images_val"] / 255.0
    c2ws_val = data["c2ws_val"]
    c2ws_test = data["c2ws_test"]
    
    if "c2ws_all" in data:
        c2ws_all = data["c2ws_all"]
    else:
        c2ws_all = c2ws_test

    print("c2ws_train shape:", c2ws_train.shape)
    print("c2ws_val shape:", c2ws_val.shape)
    print("c2ws_test shape:", c2ws_test.shape)

    H, W = images_train.shape[1:3]

    near = 0.02
    far = 6.0
    n_samples = 64

    num_iters = 1000
    batch_size = 2048
    learning_rate = 5e-4
    chunk_size = 8192

    if "K" in data:
        K = data["K"].astype(np.float32)
    else:
        focal = data["focal"]
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1.0],
        ], dtype=np.float32)

    model, stats = train_nerf(
        images_train, c2ws_train,
        images_val, c2ws_val,
        K,
        num_iters=num_iters,
        batch_size=batch_size,
        lr=learning_rate,
        num_ray_samples=n_samples,
        near=near,
        far=far,
        chunk_size=chunk_size,
        file=file_name
    )

    final_model_path = f"nerf_output/{file_name}/nerf_model_{file_name}_final.pth"
    torch.save(model.state_dict(), final_model_path)

    it_hist, val_psnr_hist, render_snapshots = stats
    plot_results(it_hist, val_psnr_hist, render_snapshots, images_val[0])

    create_gif(num_views=min(6, len(images_val)), file=file_name, duration=300)

    render_orbit_video(model, K, H, W, near, far, n_samples,
                       c2ws_all, f"nerf_output/{file_name}/test_pose_orbit_{file_name}.gif", 
                       chunk=chunk_size)

    start_pos = c2ws_train[0, :3, 3]
    render_novel_orbit(model, K, H, W, near, far, n_samples,
                       start_pos, f"nerf_output/{file_name}/novel_orbit_{file_name}.gif", 
                       num_frames=60, chunk=chunk_size)

if __name__ == "__main__":
    main()