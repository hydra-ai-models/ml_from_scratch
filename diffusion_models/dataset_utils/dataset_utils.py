import logging
import torch
from typing import Any
from IPython.display import display
from transformers import tokenization_utils_base
from torchvision import transforms

def verify_columns(dataset, image_column: str, caption_column: str) -> None:
    '''
        Verify if the dataset contains the image and caption columns, and fails if
        either the image_column or caption_column is not present in the dataset.

        Args:
            1. dataset: Dataset in HuggingFace Apache Arrow format.
            2. image_column: Name of the column containing image data.
            3. caption_column: Name of the column containing caption text data.

        Returns:
            None.
    '''
    dataset_columns = dataset.column_names
    assert image_column in dataset_columns, f'{image_column} not in {dataset_columns}.'
    assert caption_column in dataset_columns, f'{caption_column} not in {dataset_columns}.'

def visualize_dataset(dataset, image_column: str, caption_column: str, tokenizer: Any, visualize: bool = True) -> None:
    '''
        Visualize dataset by displaying length, and the first element. The image in the first element is displayed, and the
        caption text is printed.

        Args:
            1. dataset: Dataset to be visualized, in the HuggingFace Apache Arrow format.
            2. image_column: Name of the column containing image bytes.
            3. caption_column: Name of the column containing the caption text.
            4. tokenizer: Tokenizer to use for decoding text. This object should support a decode() function which takes the token index in a dictionary
                with key input_ids as input, and converts it to the corresponding text.
            5. visualize: Whether to visualize the data or not. This flag can be used to turn on or off the visualization.

        Returns:
            None.
    '''
    if not visualize:
        return

    logging.info(f'Length of dataset is {len(dataset)}.')
    assert len(dataset) > 0, 'Dataset is empty.'
    first_element = dataset[0]
    logging.info(f' First element in dataset is {first_element}.')
    image = first_element[image_column]
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    display(image)
    caption = first_element[caption_column]
    if isinstance(caption, tokenization_utils_base.BatchEncoding):
        caption = tokenizer.decode(caption["input_ids"])
    logging.info(f'First element caption is {caption}.')

def preprocess_dataset(dataset, image_size, image_column, caption_column, tokenizer, output_image_key = 'image', output_text_key = 'text'):
    '''
        Preprocess dataset to prepare it for training. This function will apply resizing and normalization of image data,
        and tokenization of the text data.

        Args:
            1. dataset: Dataset to be visualized, in the HuggingFace Apache Arrow format.
            2. image_size: Size to resize image. Image will be resized to square without preserving aspect ratio by setting width and height
                of the resized image to image_size.
            3. image_column: Name of the column containing image bytes.
            4. caption_column: Name of the column containing the caption text.
            5. tokenizer: Tokenizer to use for decoding text. This object should support a decode() function which takes the token index in a dictionary
                with key input_ids as input, and converts it to the corresponding text.
            6. output_image_key: Key of the preprocessed image data in the output. Default value is image.
            7. output_text_key: Key of the preprocessed text data in the output. Default value is text.

        Returns:
            1. A dictionary containing the processed image bytes keyed by output_image_key, and processed text data keyed by output_text_key.
    '''
    transforms_list = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    preprocessed_images = [transforms_list(i) for i in dataset[image_column]]
    tokenized_text = [tokenizer(t) for t in dataset[caption_column]]
    return {output_image_key: preprocessed_images, output_text_key: tokenized_text}

def collate_fn(data: list[Any], tokenizer: Any, image_key: str = 'image', text_key: str = 'text') -> dict[str, torch.Tensor]:
    '''
        Function which takes a list of key value pairs as input, and creates batched tensors for image and text.
        This function can be used in a Dataloader as the collate_fn input to collate data into batches.

        Args:
            1. data: List of key value pairs containing keys image_key and text_key.
            2. tokenizer: Tokenizer to use for decoding text. This object should support a pad() function which takes a list of token indices as input, and
                create a batched tensor.
            3. image_key: Key of the image data in each element in data. Default value is image.
            4. text_key: Key of the text data in each element in data. Default value is text.

        Returns:
            A dictionary of the form {image_key: Image tensor of size [Batch, Channels, Height, Width], text_key: Text tensor of size [Batches, Channels, BlockSize]}.
    '''
    output = {}
    output[image_key] = torch.stack([d[image_key] for d in data])
    output[text_key] = tokenizer.pad([d[text_key] for d in data],
                                             padding=True,
                                             return_tensors='pt')
    return output
