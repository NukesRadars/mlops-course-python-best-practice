from src.main import ImageData, ImgProcess, Predictor


def test_image_classification():
    loader = ImageData("images/")
    images = loader.load_images()
    assert len(images) == 1

    processor = ImgProcess(256)
    processed_images = processor.resize_and_gray(images)
    assert len(processed_images) == 1

    pred = Predictor()
    results = pred.predict_img(processed_images)
    assert len(results) == 1
    assert results[0] == 285  # has to predict cat
