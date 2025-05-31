import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    model = UNet2DModel.from_pretrained('your_model_path', weight_dtype=torch.float32)
    scheduler = DDPMScheduler.from_pretrained('your_scheduler_path')
    model.to(device)
    model.eval()
    return model, scheduler

model, scheduler = load_model()

# load search model
model.load_state_dict(torch.load('your_weights.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def denoise_image(img):
    def preprocess_image(img):
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        img /= 255.0
        image = Image.fromarray((img * 255).astype(np.uint8))
        original_size = image.size
        image = transform(image)
        tensor = image.unsqueeze(0).to(torch.float32).to(device)
        return tensor, original_size

    noisy_image, original_size = preprocess_image(img)

    with torch.no_grad():
        xt = noisy_image.clone()
        for t in scheduler.timesteps:
            model_output = model(xt, timestep=t).sample
            xt = scheduler.step(model_output, t, xt).prev_sample

        denoised_img = xt

    def postprocess_image(output_tensor, target_size):
        image = output_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        output_image = Image.fromarray(image).resize(target_size, Image.LANCZOS)
        return np.array(output_image).astype(np.float32)

    denoised_img = postprocess_image(denoised_img, original_size)
    return denoised_img
