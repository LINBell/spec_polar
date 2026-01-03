import random
import numpy as np
import torch
import os
from matplotlib import pyplot as plt, colors
from torch.autograd import Variable
from architecture import *
from utils import *
from option import opt


def visualize_denoised_results(model_path, image_index=2, save_dir='./spec_polo_results'):
    """
    Visualize denoised results for a given model and image index.

    Args:
        model_path (str): Path to the trained model
        image_index (int): Index of the test image to visualize (default: 2)
        save_dir (str): Directory to save visualization results (default: './spec_polo_results')
    """
    # Constants and setup
    batch_size = 15
    crop_size = 100

    # Load model
    if opt.method == 'hdnet':
        model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path).cuda()
    else:
        model = model_generator('spec_polo', opt.pretrained_model_path).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load filter matrix
    filter_path = '21_trans_matrix.npy'
    filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(
        batch_size, 1, 1, 1, 1).cuda()
    filter_matrix_test = filter_matrix_test[:, :, :, :crop_size, :crop_size]

    # Load test data
    test_data = LoadTraining_npy_spec_polo('F:/denoise_test')
    test_data = np.stack(test_data, axis=0).transpose(0, 3, 4, 1, 2)  # 5 x 512 612 4 4

    # Process test data in patches
    data_list = []
    for i in range(1, 6):
        for j in range(1, 6):
            test_gt = test_data[:, :, :, 100 * (i - 1):100 * i, 100 * (j - 1):100 * j]
            test_input = torch.tensor(test_gt).cuda()
            test_input[:, :, 0, :, :] /= opt.ratio

            with torch.no_grad():
                model_out = model(test_input, filter_matrix_test)
                model_out = model_out.view(batch_size, 21, 4, crop_size, crop_size)
            data_list.append(model_out)

    # Reconstruct full image from patches
    full_image = reconstruct_full_image(data_list)
    model_out = full_image.detach().cpu().numpy()
    real_gt = test_data[:, :, :, :500, :500]

    # Calculate PSNR
    psnr = np_psnr(model_out[image_index, :, 0, :, :],
                   real_gt[image_index, :, 0, :, :])
    print(f'PSNR: {psnr:.2f}')

    # Create save directory
    save_path = os.path.join(save_dir, f'{image_index}_{psnr:.2f}')
    os.makedirs(save_path, exist_ok=True)

    # Generate visualizations
    generate_visualizations(real_gt, model_out, image_index, save_path)

    # Analyze polarization properties
    analyze_polarization(real_gt, model_out, image_index, save_path)


def reconstruct_full_image(data_list):
    """Reconstruct full image from patches"""
    grid = [[None for _ in range(5)] for _ in range(5)]
    index = 0
    for i in range(5):
        for j in range(5):
            grid[i][j] = data_list[index]
            index += 1

    rows = []
    for i in range(5):
        row_blocks = [grid[i][j] for j in range(5)]
        concatenated_row = torch.cat(row_blocks, dim=4)  # Concatenate along width
        rows.append(concatenated_row)

    return torch.cat(rows, dim=3)  # Concatenate along height


def generate_visualizations(real_gt, model_out, index, save_path):
    """Generate various visualizations comparing GT and reconstructed images"""
    # RGB visualization
    gt_rgb = stack_21(real_gt[index, :, 0, :, :].transpose(1, 2, 0))
    out_rgb = stack_21(model_out[index, :, 0, :, :].transpose(1, 2, 0))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_rgb)
    plt.title('GT')
    plt.subplot(1, 2, 2)
    plt.imshow(out_rgb)
    plt.title('Reconstructed')
    plt.savefig(os.path.join(save_path, 'rgb_comparison.png'))
    plt.close()

    # Single channel visualization
    gt_channel = real_gt[index, 0, 0, :, :]
    out_channel = model_out[index, 0, 0, :, :]
    gt_channel = normalize_image(gt_channel)
    out_channel = normalize_image(out_channel)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_channel, cmap='gray')
    plt.title('GT (Channel 0)')
    plt.subplot(1, 2, 2)
    plt.imshow(out_channel, cmap='gray')
    plt.title('Reconstructed (Channel 0)')
    plt.savefig(os.path.join(save_path, 'channel0_comparison.png'))
    plt.close()

    # Polarization component visualization
    gt_pol = real_gt[index, 0, 3, :, :]
    out_pol = model_out[index, 0, 3, :, :]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_pol, cmap='bwr', vmin=-np.max(np.abs(gt_pol)),
               vmax=np.max(np.abs(gt_pol)))
    plt.colorbar()
    plt.title('GT Polarization Component')
    plt.subplot(1, 2, 2)
    plt.imshow(out_pol, cmap='bwr', vmin=-np.max(np.abs(out_pol)),
               vmax=np.max(np.abs(out_pol)))
    plt.colorbar()
    plt.title('Reconstructed Polarization Component')
    plt.savefig(os.path.join(save_path, 'polarization_component.png'))
    plt.close()

    # Spectral plots
    for i in range(3, 11):
        select_h = random.randint(0, 500)
        select_w = random.randint(0, 500)
        gt_spectrum = real_gt[index, :, 0, select_h, select_w]
        out_spectrum = model_out[index, :, 0, select_h, select_w]

        plt.figure()
        plt.plot(range(21), gt_spectrum, label='GT', marker='o', color='red')
        plt.plot(range(21), out_spectrum, label='Reconstructed', marker='s', color='blue')
        plt.xlabel('Spectral Band')
        plt.ylabel('Intensity')
        plt.legend()
        plt.title(f'Spectral Profile at ({select_h}, {select_w})')
        plt.savefig(os.path.join(save_path, f'spectral_profile_{i}.png'))
        plt.close()


def analyze_polarization(real_gt, model_out, index, save_path):
    """Analyze and visualize polarization properties"""
    # Check Stokes parameters constraints
    gt_stokes = real_gt[index, :, :, :, :].transpose(2, 3, 0, 1)
    out_stokes = model_out[index, :, :, :, :].transpose(2, 3, 0, 1)

    print('GT Stokes Parameters:')
    cal_restric(gt_stokes)
    print('Reconstructed Stokes Parameters:')
    cal_restric(out_stokes)

    # Polarization ellipse visualization
    gt_pixel = real_gt[index, 0, :, 250, 250]
    out_pixel = model_out[index, 0, :, 250, 250]

    plt.figure()
    plot_polarization_ellipse(gt_pixel[1], gt_pixel[2], gt_pixel[3])
    plt.title("GT Polarization Ellipse")
    plt.savefig(os.path.join(save_path, 'gt_polarization_ellipse.png'))
    plt.close()

    plt.figure()
    plot_polarization_ellipse(out_pixel[1], out_pixel[2], out_pixel[3])
    plt.title("Reconstructed Polarization Ellipse")
    plt.savefig(os.path.join(save_path, 'recon_polarization_ellipse.png'))
    plt.close()

    # Polarization maps
    plot_polarization_maps(gt_stokes, "GT", os.path.join(save_path, 'gt_polarization_maps.png'))
    plot_polarization_maps(out_stokes, "Reconstructed", os.path.join(save_path, 'recon_polarization_maps.png'))


# Helper functions
def stack_21(img):
    """Stack specific bands to create RGB image"""
    band1 = img[:, :, 2]
    band2 = img[:, :, 10]
    band3 = img[:, :, 18]
    return np.stack([band1, band2, band3], axis=-1)


def cal_restric(data):
    """Check Stokes parameters constraints"""
    s0 = data[..., 0]
    s1 = data[..., 1]
    s2 = data[..., 2]
    s3 = data[..., 3]

    left_side = s0 ** 2
    right_side = s1 ** 2 + s2 ** 2 + s3 ** 2
    condition_met = left_side >= right_side

    total_points = np.prod(data.shape[:-1])
    valid_points = np.sum(condition_met)
    invalid_points = total_points - valid_points

    print(f"Total points: {total_points}")
    print(f"Valid points (S0² ≥ S1²+S2²+S3²): {valid_points} ({valid_points / total_points:.2%})")
    print(f"Invalid points: {invalid_points} ({invalid_points / total_points:.2%})")


def np_psnr(img, ref):
    """Calculate PSNR between images"""
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = np.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * np.log10((255 * 255) / mse)
    return psnr / nC


def plot_polarization_ellipse(S1, S2, S3, scale=1):
    """Plot polarization ellipse for given Stokes parameters"""
    psi = 0.5 * np.arctan2(S2, S1)
    DoP = np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2)
    chi = 0.5 * np.arcsin(S3 / (DoP + 1e-9))

    a = DoP * scale
    b = a * np.tan(chi)

    theta = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(theta) * np.cos(psi) - b * np.sin(theta) * np.sin(psi)
    y = a * np.cos(theta) * np.sin(psi) + b * np.sin(theta) * np.cos(psi)

    plt.plot(x, y, 'r-', linewidth=1)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
    plt.axis('equal')


def plot_polarization_maps(stokes_data, title, save_path):
    """Plot polarization azimuth and degree maps"""
    # Ensure we're only processing one channel (first channel)
    S0 = stokes_data[..., 0, 0]  # Shape: (H, W)
    S1 = stokes_data[..., 0, 1]  # Shape: (H, W)
    S2 = stokes_data[..., 0, 2]  # Shape: (H, W)
    S3 = stokes_data[..., 0, 3]  # Shape: (H, W)

    azimuth_deg = 0.5 * np.rad2deg(np.arctan2(S2, S1))
    DoP = np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / (S0 + 1e-10)

    plt.figure(figsize=(12, 5))

    # 1. Azimuth Angle (HSV)
    plt.subplot(1, 2, 1)
    hue = (azimuth_deg + 90) / 180.0  # Normalize to [0,1]
    saturation = np.ones_like(hue)
    value = np.ones_like(hue)
    hsv_image = np.stack([hue, saturation, value], axis=-1)
    rgb_image = colors.hsv_to_rgb(hsv_image)
    plt.imshow(rgb_image)
    plt.title(f"{title} Azimuth Angle")
    plt.colorbar(ticks=[-90, -45, 0, 45, 90], label="Angle (°)")

    # 2. Degree of Polarization
    plt.subplot(1, 2, 2)
    plt.imshow(DoP, cmap='gray', vmin=0, vmax=1)
    plt.title(f"{title} Degree of Polarization")
    plt.colorbar(label="DoP (0~1)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_polarization(real_gt, model_out, index, save_path):
    """Analyze and visualize polarization properties"""
    # Select the first spectral channel (index 0)
    gt_stokes = real_gt[index, 0, :, :, :].transpose(1, 2, 0)  # Shape: (H, W, 4)
    out_stokes = model_out[index, 0, :, :, :].transpose(1, 2, 0)  # Shape: (H, W, 4)

    print('GT Stokes Parameters:')
    cal_restric(np.expand_dims(gt_stokes, axis=0))  # Add batch dim for compatibility
    print('Reconstructed Stokes Parameters:')
    cal_restric(np.expand_dims(out_stokes, axis=0))

    # Polarization ellipse visualization
    gt_pixel = gt_stokes[250, 250, :]  # Shape: (4,)
    out_pixel = out_stokes[250, 250, :]

    plt.figure()
    plot_polarization_ellipse(gt_pixel[1], gt_pixel[2], gt_pixel[3])
    plt.title("GT Polarization Ellipse")
    plt.savefig(os.path.join(save_path, 'gt_polarization_ellipse.png'))
    plt.close()

    plt.figure()
    plot_polarization_ellipse(out_pixel[1], out_pixel[2], out_pixel[3])
    plt.title("Reconstructed Polarization Ellipse")
    plt.savefig(os.path.join(save_path, 'recon_polarization_ellipse.png'))
    plt.close()

    # Polarization maps
    plot_polarization_maps(np.expand_dims(gt_stokes, axis=2),  # Add channel dim: (H, W, 1, 4)
                           "GT",
                           os.path.join(save_path, 'gt_polarization_maps.png'))
    plot_polarization_maps(np.expand_dims(out_stokes, axis=2),
                           "Reconstructed",
                           os.path.join(save_path, 'recon_polarization_maps.png'))


def normalize_image(img):
    """Normalize image to [0, 1] range"""
    return (img - np.min(img)) / (np.max(img) - np.min(img))


visualize_denoised_results(
    model_path='polo_test_model/denoise_slice_87_30.pth',
    image_index=0,
    save_dir='C:\\Users\\38362\\Desktop\\spec_polo_pic'
)