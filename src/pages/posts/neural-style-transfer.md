---
layout: '@/templates/BasePost.astro'
title: Neural style transfer
description: This article explains neural style transfer, an AI technique combining the visual content of one image with the artistic style of another. It details how convolutional neural networks capture content and style, and how iterative optimization blends the two into a new hybrid image. A clear guide to this generative deep learning approach.
pubDate: 2023-09-03
imgSrc: '/assets/images/neural-style-transfer/index.jpg'
imgAlt: "Vladimir Putin's photo modified by artificial intelligence with an aquarel style"
---

# Introduction
To generally separate content from style in natural images is still an extremely difficult problem. However, the recent advance of Deep Convolutional Neural Networks has produced powerful computer vision systems that learn to extract high-level semantic information from natural images.
Transferring the style from one image onto another can be considered a problem of texture transfer. In texture transfer, the goal is to synthesise a texture from a source image while constraining the texture synthesis in order to preserve the semantic content of a target image.
For texture synthesis, there exist a large range of powerful non-parametric algorithms that can synthesise photorealistic natural textures by resampling the pixels of a given source texture. Therefore, a fundamental prerequisite is to find image representations that independently model variations in the semantic image content and the style in which is presented.

![Style Neural Network results](/assets/images/neural-style-transfer/styleNeuralNetResult1.png)

As we can see, the generated image is having the content of the ***Content image and style image***. This above result cannot be obtained by overlapping the image. So the main question are:  ***What is neural style transfer? how we make sure that the generated image has the content and style of the image?  how we capture the content and style of respective images?***

# What is neural style transfer?

Neural Style Transfer is the technique of blending style from one image into another image keeping its content intact. The only change is the style configurations of the image to give an artistic touch to your image.
The content image describes the layout or the sketch and Style being the painting or the colors. Neural Style Transfer deals with two sets of images: **Content image and Style image**.
This technique helps to recreate the content image in the style of the reference image. It uses Neural Networks to apply the artistic style from one image to another.
Neural style transfer opens up endless possibilities in design, content generation, and the development of creative tools.


# How does Style Transfer work?

The goal of Neural Style Transfer (NST) is to give the deep learning model the ability to differentiate between style representations and content images. NST uses a pre-trained convolutional neural network with additional loss functions to transfer the style from one image to another and synthesize a newly generated image with the desired features.
Style transfer works by activating the neurons in a specific way, so that the output image and the content image match particularly at the content level, while the style image and the desired output image should match in terms of texture and capture the same style characteristics in the activation maps.
These two objectives are combined in a single loss formula, where we can control how much we care about style reconstruction and content reconstruction.

The **loss function** in neural style transfer plays a crucial role in guiding the model to generate an image that combines both the desired style and content. In style transfer, we typically have a content image (e.g., a photograph) and a style image (e.g., an artwork). The goal is to generate a new image that preserves the content of the content image while adopting the style of the style image.
The loss function is used to quantify how well the generated image matches the style and content objectives. It measures the difference between the network activations for the original content image and the generated image, as well as the difference between the network activations for the original style image and the generated image.

To balance style reconstruction and content reconstruction, the loss function combines these two differences using weights. These weights control the relative importance of style reconstruction compared to content reconstruction.
Optimizing the loss function involves adjusting the pixel values in the generated image to minimize the difference between the network activations for the original content image and the generated image, while also minimizing the difference between the network activations for the original style image and the generated image.

By adjusting these pixel values, the model learns to reproduce the desired style characteristics in the corresponding areas of the generated image while preserving the content information from the original image.
Thus, the loss function optimizes the model to generate an image that combines the desired style and content by minimizing the discrepancies between the network activations for the reference images and the generated image.

Here are the required inputs to the model for image style transfer:

1. **A Content Image** –an image to which we want to transfer style to
2. **A Style Image** – the style we want to transfer to the content image
3. **An Input Image** (generated) – the final blend of content and style image


## Neural Style Transfer basic structure

Training a style transfer model requires two networks: **a pre-trained feature extractor and a transfer network**.

In the case of Neural Style Transfer (NST), we use a model pre-trained on ImageNet, such as VGG in TensorFlow. However, the VGG model cannot understand images directly. It is necessary to convert the images into raw pixels and feed them to the model to transform them into a set of features, which is usually done by convolutional neural networks (CNN) which we will see in the next section.

Thus, the VGG model acts as a complex feature extractor between the input layer, where the image is fed, and the output layer, which produces the final result. To achieve style transfer, we focus on the middle layers of the model that capture essential information about the content and style of the input images.
During the style transfer process, the input image is transformed into representations that emphasize image content rather than specific pixel values.

Features extracted from upper layers of the model are more closely related to the image content. To obtain a representation of the style from a reference image, we analyze the correlation between the different filter responses in the model.


### How Convolutional Neural Network capture features in VGG model?

![Style Neural Network results](/assets/images/neural-style-transfer/CNN_architecture.png)

The VGG model is actually a type of convolutional neural network (CNN). VGG, which stands for **Visual Geometry Group**, is a very popular CNN architecture widely used in computer vision tasks, especially in the field of image classification. The VGG model is composed of several stacked convolutional layers, followed by fully connected layers. These convolutional layers are responsible for extracting visual features from images.
Specifically, VGG's convolutional layers are designed to analyze visual patterns at different spatial scales. Each convolutional layer uses filters that are applied to images to detect specific patterns, such as edges, textures or shapes.

The first convolutional layers of VGG (those at level 1 with 32 filters) capture low-level features, such as simple edges and textures, while the deeper convolutional layers (those at level 2 with 64 filters) capture features of higher level like complex shapes and overall structures.

Thus the VGG model as a CNN is able to extract meaningful visual features from images. These features can then be used in different tasks, such as image classification or style transfer. In the context of style transfer, the model is used primarily as a feature extractor. VGG's convolutional layers are leveraged to capture content and style information from input images, allowing these two aspects to be separated and combined to generate a new image that combines the content of a reference image and style from another image.

### Content loss

The **content loss** is a measure that allows establishing similarities between the content image and the image generated by the style transfer model. The underlying idea is that the upper layers of the model focus more on the features present in the image, i.e. on the overall content of the image.
The content loss is calculated using the Euclidean distance between the upper-level intermediate feature representations of the input image (x) and the content image (p) at layer l.

$L_{content}(\vec{p},\vec{x},l) = \frac{1}{2}\sum_{i,j}(F_{ij}^{l}(x) - P_{ij}^{l}(p))^2$

In this equation, $F_{ij}^{l}(x)$ represents the feature representation of the input image x at layer l and $P_{ij}^{l}(p)$ represents the representation of characteristics of the content image p at layer l.

### Style loss


![Style Neural Network results](/assets/images/neural-style-transfer/styleLoss.png)

**Style loss** is conceptually different from content loss, but it is not possible to simply compare the intermediate features (textured patterns, contours, shapes) of the two images to obtain the style loss. That is why we introduce a new term called "Gram matrix".

In calculating content loss, the position of each pixel is taken into account in order to reproduce the content of the original image in the generated image. However, this is not the case for style loss because it relates to texture, colors, and other aspects. Thus, the Gram matrix is used to capture the stylistic features present in an image.

The initial layers of a neural network tend to capture features such as color and texture. One might think that by using a similar approach to content loss, focusing on the initial layers, we could obtain a "style loss". Unfortunately, this is not the case.

The feature maps of the initial layers not only show which features are present, but they also record where these features are located in the image. This is where the Gram matrix comes into play. It allows for the removal of the spatial component and focuses solely on the types of features present. Removing the spatial component in the Gram matrix enables us to concentrate solely on the types of features present in the image, regardless of their specific spatial location.

In the context of style loss, the goal is to capture global patterns and textures rather than focusing on the local details of the image. By eliminating the spatial component, we obtain a representation of the style features that emphasizes the relationships between different features without being influenced by their position in the image.

This enables us to compare style features between two images in a more abstract and general way, disregarding specific spatial variations. As a result, we can evaluate the stylistic similarity between images based on global patterns and textures rather than precise details of their positioning.

Furthermore, by removing the spatial component, the Gram matrix becomes a compact representation of style information, making style loss calculation easier and reducing memory and computation requirements.

Style loss is calculated by the distance between the gram matrices (or, in other terms, style representation) of the generated image and the style reference image.

The contribution of each layer in the style information is calculated by the formula below:
$E_l=\frac{1}{4 N_l^2 M_l^2} \sum_{i, j}\left(G_{i j}^l-A_{i j}^l\right)^2$

Thus, the total style loss across each layer is expressed as:
$L_{\text {style }}(a, x)=\sum_{l \in L} w_l E_l$


### Model architecture of neural style transfer

The architecture of the NST can be designed in such a way that it can range from applying a single style in an image to allowing mix and match of multiple styles.
Let’s have a look at the different possibilities.


![Style Neural Network results](/assets/images/neural-style-transfer/neuralArchitecture.png)

### Total Loss

The total loss function is the sum of the cost of the content and the style image. Mathematically,it can be expressed as :

$L_{total}(\vec{p}, \vec{\alpha},\vec{x}) = \alpha L_{content}(\vec{p}, \vec{x}) + \beta L_{style}(\vec{\alpha}, \vec{x})$

You may have noticed Alpha and beta in the above equation.They are used for weighing Content and Style cost respectively.In general,they define the weightage of each cost in the Generated output image.

Once the loss is calculated,then this loss can be minimized using **backpropagation** which in turn will optimize our **randomly generated image** into a **meaningful piece of art**.
# Conclusion

In conclusion, this article has explored the fundamental principle behind neural style transfer and its ability to successfully combine the content of one image with the style of another.
We began by understanding how convolutional neural networks progressively learn to represent image features through the extraction of simple to complex patterns. This encoding capability is crucial for separating content and style in an unsupervised manner.
Precise definitions of content and style losses were provided, mathematically modeling the desired similarity between the generated image and the reference images. The calculation of the Gram matrix emerged as central to quantifying correlations between filters indicative of style.
Optimizing the generated image through backpropagation of the overall cost function gradient then allows synthesizing a new image realistically fusing the targeted characteristics.
While still improvable, this unsupervised approach opens promising avenues for numerous artistic and industrial applications. Future work could focus on enhancing its capacities.
Through this article, we aimed to provide a clear and detailed explanation of the internal workings behind neural style transfer, foundational to many deep learning innovations. Continued research in this area holds potential for generating increasingly convincing and customized artistic blends.