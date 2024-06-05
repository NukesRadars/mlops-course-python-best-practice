import os
from PIL import Image
import torch
from torchvision import transforms, models  # type: ignore
from torchvision.models.resnet import ResNet18_Weights  # type: ignore


class ImageData:
    """
    A class to load images from a directory.

    Attributes:
        dir (str): The directory containing the images.

    Methods:
        load_images() -> list[Image.Image]:
            Loads images from the specified directory and returns a list of PIL Image objects.
    """

    def __init__(self, dir: str) -> None:
        """
        Constructs all the necessary attributes for the ImageData object.

        Parameters:
            dir (str): The directory containing the images.
        """
        self.dir = dir

    def load_images(self) -> list[Image.Image]:
        """
        Loads images from the specified directory.

        Returns:
            list[Image.Image]: A list of loaded PIL Image objects.
        """
        imgs = []
        for filename in os.listdir(self.dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                imgs.append(Image.open(os.path.join(self.dir, filename)))
        return imgs


class ImgProcess:
    """
    A class to process images by resizing and converting to grayscale.

    Attributes:
        size (int): The size to which the images will be resized.

    Methods:
        resize_and_gray(img_list: list[Image.Image]) -> list[torch.Tensor]:
            Resizes and converts a list of PIL Image objects to grayscale and returns a list of PyTorch tensors.
    """

    def __init__(self, size: int) -> None:
        """
        Constructs all the necessary attributes for the ImgProcess object.

        Parameters:
            size (int): The size to which the images will be resized.
        """
        self.size = size

    def resize_and_gray(self, img_list: list) -> list[torch.Tensor]:
        """
        Resizes and converts a list of PIL Image objects to grayscale.

        Parameters:
            img_list (list[Image.Image]): A list of PIL Image objects to be processed.

        Returns:
            list[torch.Tensor]: A list of processed PyTorch tensors.
        """
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
    """
    A class to predict labels for processed images using a pre-trained ResNet18 model.

    Attributes:
        mdl (torchvision.models.ResNet): The pre-trained ResNet18 model.

    Methods:
        predict_img(processed_images: list[torch.Tensor]) -> list[int]:
            Predicts labels for a list of processed images.
    """

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes for the Predictor object and loads the pre-trained ResNet18 model.
        """
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def predict_img(self, processed_images: list[torch.Tensor]) -> list[int]:
        """
        Predicts labels for a list of processed images.

        Parameters:
            processed_images (list[torch.Tensor]): A list of processed PyTorch tensors.

        Returns:
            list[int]: A list of predicted labels for the images.
        """
        results = []
        for img_tensor in processed_images:
            pred = self.mdl(img_tensor.unsqueeze(0))
            results.append(int(torch.argmax(pred, dim=1).item()))
        return results


if __name__ == "__main__":
    loader = ImageData("images/")
    images = loader.load_images()

    processor = ImgProcess(256)
    processed_images = processor.resize_and_gray(images)

    pred = Predictor()
    results = pred.predict_img(processed_images)
    print(results)
