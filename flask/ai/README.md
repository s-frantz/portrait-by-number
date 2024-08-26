# **AI Models**

Basic computer vision algorithms perform OK at foreground-background segmentation tasks, but commonly mix up the two categories when contexts are similar or regions of either category are surrounded by the other.

### **Portrait segmentation**

Relatively more specific models exist for segmenting various sub-categories of features, mostly building on UNET models originally developed for medical image segmentation.

We now have very strong models for segmenting portraits, and many of these are available on GitHub [e.g., foreground-background](https://github.com/by321/me2net/blob/main/readme.md) & [selfie segmentation](https://mediapipe-studio.webapps.google.com/demo/image_segmenter). The latter is interesting -- a Google mediapipe model in TensorFlow, which differentiates face / body / hair / hat / clothing quickly and with good results.

The model we take on board in this project is comparable, but comes from a face-segmentation [project](https://github.com/zllrunning/face-parsing.PyTorch), which segments portraits into 18 classes, going as granular as r_brow, l_brow (right eyebrow, left eyebrow) and u_lip, l_lip (upper lip, lower lip).

For our purposes, we are interested in just a few of these, specifically, we want to differentiate and choose appropriate color spaces for:

1. the face (incl. eyes, ears, mouth, nose, etc.)
2. face framing features (hair, hat, neck)
3. clothing
4. background

The more detailed a paint-by-number gets, the more time the user has to invest in completing it, and the more true to the original the result, we want to ensure that paint-by-number blocks preferrentially assign the most detail to 1., and the least to 4., which may "focus" and "blur" these classes relatively.

We also want to ensure that paint-by-number blocks do not bleed across our distinct classes. Even if face & hair or hair & clothing look locally similar, we stipulate that a boundary should exist between them.

Below is the full list of features outputted by the deep learning model, and which we are able to group for the paint-by-number:

 - 1 'skin',
 - 2 'l_brow',
 - 3 'r_brow',
 - 4 'l_eye',
 - 5 'r_eye',
 - 6 'eye_g',
 - 7 'l_ear',
 - 8 'r_ear',
 - 9 'ear_r',
 - 10 'nose',
 - 11 'mouth',
 - 12 'u_lip',
 - 13 'l_lip',
 - 14 'neck',
 - 15 'neck_l',
 - 16 'cloth',
 - 17 'hair',
 - 18 'hat'