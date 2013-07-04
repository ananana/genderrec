genderrec
=========

Recognizing the gender of a person using the image of his/her face.

The project uses the computer vision library openCV (with C++) and Qt to classify face images as male/female. Three different techniques were implemented for training and classifying - eigenfaces, fisherfaces, and PCA+SVM. Fisherfaces which gave the best results (95% accuracy) is used by default.

The application can use either a static picture, or capture from the webcam to detect and classify the faces found in the image / in each frame of the video.
