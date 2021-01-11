# ViTSpeedPrediction


The goal of this project is to take on the comma.ai speed classification challenge while also exploring some new developments in model archictectures over the past year. Specifically, I would like to try to apply the novel Vision Transformer (ViT) architecture in learning to speed classification. 


To solve the classification challenge, I need to label each frame so I want to feed in image data to ViT. However since the point is to detect how fast the vehicle is moving from the camera input, I need to encode some information about how the images are changing so the model can predict speed from that. Fortunately, there's a very good way to encode this information called Optical Flow. 


Optical Flow is a measure of the apparent velocity of objects in a video. There are a few ways to calculate this. There's an OpenCV library available to calculate dense optical flow and this was my first approach. However, I later learned of a cool development out of Princeton this year - Recurrent All Pairs Field Transforms for Optical Flow (RAFT). I don't really have the bandwidth to train a full RAFT model, but they have pre-trained examples which are SOTA and faster and more precise than the OpenCV method. Since I am pre-processing the data, it doesn't really matter to me, but if I was to use this to drive a vehicle, then speed would be paramount. 


Conceptually, I am hoping that the Vision Transformer will be able to focus its attention selectively on the parts of the images that correspond to objects whose velocities best indicate the speed of the car. Then the model will learn to estimate the speed of the vehicle from these velocities and hopefully outperform current methods.

