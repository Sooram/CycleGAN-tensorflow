# CycleGAN-tensorflow
CycleGAN implemented with tensorflow

## Reference 
https://github.com/leehomyc/cyclegan-1 \
https://github.com/xhujoy/CycleGAN-tensorflow \
https://github.com/vanhuyz/CycleGAN-TensorFlow \
https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d

## Original paper
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
![Overview](https://github.com/Sooram/CycleGAN-tensorflow/blob/master/imgs/overview.PNG)

## Loss function
1. Adversarial Loss\
![Overview](https://github.com/Sooram/CycleGAN-tensorflow/blob/master/imgs/adversarial-loss.PNG)

2. Cycle Consistency Loss\
![Overview](https://github.com/Sooram/CycleGAN-tensorflow/blob/master/imgs/cycle-consistency-loss.PNG)\
"adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi. To further reduce the space of possible mapping functions, we argue that the learned mapping functions should be cycle-consistent
 for each image x from domain X, the image translation cycle should be able to bring x back to the original image, i.e., x → G(x) → F(G(x)) ≈ x."\
![Overview](https://github.com/Sooram/CycleGAN-tensorflow/blob/master/imgs/cycle-consistency-loss-img.PNG)

3. Full Objective\
![Overview](https://github.com/Sooram/CycleGAN-tensorflow/blob/master/imgs/full-objective.PNG)

## Network architecture
Pretty complicated and a lot of restrictions. I haven't implemented PatchGAN for the discriminator.
1. Generator\
![Overview](https://github.com/Sooram/CycleGAN-tensorflow/blob/master/imgs/generator.PNG)\
"This network contains two stride-2 convolutions, several residual blocks [18], and two fractionally strided convolutions with stride 1 2 . We use 6 blocks for 128 × 128 images and 9 blocks for 256 × 256 and higher resolution training images. Similar to Johnson et al. [23], we use instance normalization [53]."

2. Discriminator\
![Overview](https://github.com/Sooram/CycleGAN-tensorflow/blob/master/imgs/discriminator.PNG)\
"For the discriminator networks we use 70 × 70 PatchGANs [22, 30, 29], which aim to classify whether 70 × 70 overlapping image patches are real or fake."

3. Training details
- use a least-square loss instead of the negative log likelihood objective for equation 1
- image pool size: 50 (update discriminators using 50 previously generated images to reduce model oscillation)
- λ = 10 in Equation 3
- optimizaer: Adam solver
- batch size: 1
- learning rate: 0.0002 for the first 100 epochs and linearly decay the rate to zero over the next 100 epochs

## Results
Will be updated soon.
