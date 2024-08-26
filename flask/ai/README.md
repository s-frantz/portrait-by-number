## **AI Models**

### **Overview**

Pre-deep-learning computer vision algorithms have middle-of-the-road performance when it comes to foreground-background segmenting tasks. They typically rely on a combination of existing techniques like clustering and canny edge detection, but commonly [mix up categories](https://medium.com/@muhammadsabih56/background-subtraction-in-computer-vision-402ddc79cb1b) when contexts are similar, or regions of one category are entirely surrounded by the other.

With the advent of deep learning, many more accurate models have now been written for specific use-cases, primarily building on the original UNET model for medical image segmentation.


### **Portrait segmentation**

#### **Options**

One such use-case of interest to this project is for segmenting (esp. portrait-style) photos of people. The source code for such models are available on GitHub, including a few we looked at:

 1. [foreground-background](https://github.com/by321/me2net/blob/main/readme.md)
 2. [selfie segmentation](https://mediapipe-studio.webapps.google.com/demo/image_segmenter)
 3. [face segmentation](https://github.com/zllrunning/face-parsing.PyTorch)
 4. [face segmentation](https://huggingface.co/jonathandinu/face-parsing)

Though the second of these, a Google media-pipe model written in TensorFlow, is quite interesting and able to segment `face`, `body`, `hair`, `hat`, `clothing`, our project opts to take on board the third, which segments portraits into 18 unique classes, as granular as `r_brow`, `l_brow` (right eyebrow, left eyebrow) and `u_lip`, `l_lip` (upper lip, lower lip). The fourth project listed is extremely similar and may be a copy of #3 or vice versa.

#### **Application**

Not all of these are interesting as paint-by-number super-classes, but we want to differentiate:

1. the face (eyes, ears, mouth, nose, etc.)
2. face framing features (hair, hat, neck)
3. clothing
4. background

Separating these allows for using custom color-spaces (e.g., for skin tones), and for providing more detail in higher classes. The paint-time for an artist / puzzler will be determined largely by the number and complexity of paint-by-number blocks, as will, inversely, the quality of that end result. Focusing paint-time and resolution on higher-priority image regions will give our users a better ROI.

We also want to ensure that paint-by-number blocks do not bleed across these classes. Even if face & hair or hair & clothing look locally similar, a boundary should exist between them.

Below is the full list of features outputted by the deep learning model:

 - 1 'skin', #the face
 - 2 'l_brow', #left eyebrow
 - 3 'r_brow', #right eyebrow
 - 4 'l_eye', #left eye
 - 5 'r_eye', #right eye
 - 6 'eye_g', #eye glasses
 - 7 'l_ear', #left ear
 - 8 'r_ear', #right ear
 - 9 'ear_r', #ear ring
 - 10 'nose', #nose
 - 11 'mouth', #mouth
 - 12 'u_lip', #upper lip
 - 13 'l_lip', #lower lip
 - 14 'neck', #neck
 - 15 'neck_l', #necklace
 - 16 'cloth', #clothing
 - 17 'hair', #hair
 - 18 'hat' #hat