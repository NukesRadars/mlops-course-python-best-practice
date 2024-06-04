import os
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights


class ImageData:
    def __init__(self, dir):
        self.dir = dir

    def load_images(self):
        imgs = []
        for filename in os.listdir(self.dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                imgs.append(Image.open(os.path.join(self.dir, filename)))
        return imgs


class ImgProcess:
    def __init__(self, size):
        self.size = size

    def resize_and_gray(self, img_list):
        p_images = []
        for img in img_list:
            t = transforms.Compose(
                [
                    transforms.Resize((self.size, self.size)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            p_images.append(t(img))
        return p_images


class Predictor:
    def __init__(self):
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def predict_img(self, processed_images):
        results = []
        for img_tensor in processed_images:
            pred = self.mdl(img_tensor.unsqueeze(0))
            results.append(torch.argmax(pred, dim=1).item())
        return results


if __name__ == "__main__":
    loader = ImageData("images/")
    images = loader.load_images()

    processor = ImgProcess(256)
    processed_images = processor.resize_and_gray(images)

    pred = Predictor()
    results = pred.predict_img(processed_images)
    print(results)
