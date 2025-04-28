import os
import numpy as np
import matplotlib.pyplot as plt

def convert_to_grid(x_input):
    if x_input.shape[-1] not in [1, 3]:
        x_input = np.transpose(x_input, (0, 1, 3, 2))

    N, H, W, C = x_input.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + 1 * (grid_size - 1)
    grid_width = W * grid_size + 1 * (grid_size - 1)

    grid = np.ones((grid_height, grid_width, 3)) * 255

    next_idx = 0
    y0 = 0
    for y in range(grid_size):
        x0 = 0
        for x in range(grid_size):
            if next_idx < N:
                img = x_input[next_idx]
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                elif img.shape[-1] == 4:
                    img = img[..., :3]

                norm_img = 255 * (img - img.min()) / (img.max() - img.min() + 1e-8)

                grid[y0:y0+H, x0:x0+W, :] = norm_img.reshape(H, W, 3)
                next_idx += 1
            x0 += W + 1
        y0 += H + 1

    return grid.astype('uint8')

def visualize_grid_samples(data, number_of_samples=64, save_path='./figures/samples.png'):
    examples = data['x_train'][:number_of_samples]
    print("Original shape:", examples.shape)

    plt.figure(figsize=(15, 15))
    plt.imshow(convert_to_grid(examples))
    plt.axis('off')
    plt.title(f'Traffic Sign Images (Training): {number_of_samples}', fontsize=18)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
