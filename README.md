Fine-tuning DETR on a custom dataset for object detection
In this notebook, we are going to fine-tune DETR (end-to-end object detection with Transformers) on a custom object detection dataset. The goal for the model is to detect balloons in pictures.

Original DETR paper: https://arxiv.org/abs/2005.12872
Original DETR repo: https://github.com/facebookresearch/detr
Note regarding GPU memory
DetrImageProcessor by default resizes each image to have a min_size of 800 pixels and a max_size of 1333 pixels (as these are the default values that DETR uses at inference time). Note that this can stress-test the memory of your GPU when training the model, as the images are flattened after sent through the convolutional backbone. The sequence length that is sent through the Transformer is typically of length (height*width/32^2). So if you consider an image of size (900, 900) for example, the sequence length is 900^2/32^2 = 791, which is larger than what NLP models like BERT use (512). It's advised to use a batch size of 2 on a single GPU. You can of course also initialize DetrImageProcessor with a smaller size and/or max_size to use bigger batches.

Note regarding data augmentation
DETR actually uses several image augmentations during training. One of them is scale augmentation: they set the min_size randomly to be one of [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800] as can be seen here. However, we are not going to add any of the augmentations that are used in the original implementation during training. It works fine without them.

Training framework
We're going to fine-tune the model using PyTorch Lightning, but of course you could also train the model using native PyTorch, the 🤗 Trainer class, 🤗 Accelerate, or any other framework you prefer.

Also big thanks to the creator of this notebook, which helped me a lot in understanding how to fine-tune DETR on a custom dataset.
Dataset link-https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg
