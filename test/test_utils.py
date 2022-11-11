from utils.entropy_loss import get_guassian_heatmaps_from_ref
from tutils import torchvision_save
import torch


def test_gaussian_map():
    landmarks = [[128,128]]
    size = (384,384)
    kernel_size = 224
    sharpness=05
    prob_maps = get_guassian_heatmaps_from_ref(landmarks=landmarks, num_classes=len(landmarks), \
                                           image_shape=size, kernel_size=kernel_size,
                                           sharpness=sharpness)  # shape: (19, 800, 640)
    prob_maps = torch.Tensor(prob_maps)
    torchvision_save(prob_maps, f"prob_ks{kernel_size}_shp{sharpness}.jpg")

if __name__ == '__main__':
    test_gaussian_map()