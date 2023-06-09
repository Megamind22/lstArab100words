# Deep Visual Speech Recognition  for lstArab100words

Our application works to support Visual Speech Recognition (VSR) is 
the ability to recognize words from the mouth movements of a speaking 
person Lip reading in arabic language,the system using a locally collected dataset that 
was prepared and photographed through us, and the number of videos 
(9000)

# Requirements
<h4>fastapi<h4/>
<h4>uvicorn<h4/>
<h4>pickle5<h4/>
<h4>pydantic<h4/>
<h4>scikit-learn<h4/>
<h4>requests<h4/>
<h4>pypi-json<h4/>
<h4>pyngrok==4.1.1<h4/>

# Project Structure
#### Files

# We train the model using two types of dataset
#### YouTube dataset
  
In the beginning, we will talk about the first approach to building our owndatabase, which is by collecting videos from YouTube each video isbetween 3 and 7 seconds long and 25 fps. We take from multiple channel ex sada Elbalad and different platforms about 50 people (20males an 30 females), And we have collected 6,000 videos, and it isentered in several stages to be prepared for processing.

The multiple stages are:
1- We upload the video in 360 or 240 qualities.
  
2- Then we extract text from video
  
3- Then we make a time stamp in which every word in video said in certain amount of seconds (00:00 00:02 "اخبار")With using tool named “time thing”

#### Camera dataset
  
Now, we will talk about the second approach to building our database,where in this approach we photographed people with our mobilecamera. 9000 videos were captured for different participants from both genders (i.e. 24 females and 62 males). Age range is 13 to 75 years from the same country. . The videos of the dataset are captured with different background lighting settings, different distances, and using cameras of multismartphones.
Different cameras, distances, and background lighting settings guarantee dataset generalization. All used cameras record videos at a rate of 30 frames per second (fps). All 86 speakers uttered each one of the 100 words once (i.e. 9000 videos)
This ensures the inclusiveness of the dataset because there are many diversities between people that need to be accounted for, such as: Speediness of speaking, mouth shape and movements, lips geometric features, amount of tongue determined by the redness of the taken mouth frame, the alveolar ridge, teeth, braces, mustache, beard, and makeup.


# Preprocessing

One of the most important steps in our application tasks is to preprocess and split the dataset before feeding it to the learning algorithm. In our application, the following preprocessing steps are required:
1) The lengths of all videos in the dataset are fixed to exactly one or two second while making sure that each video contains the target word without any errors.
2) Each video is converted to 30 frames of images using the Video Capture class from the Python cv2 library 
3) Each frame is processed using Face Tracker Which tracks the movement of the person's face in the video. You can find more details in https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/tree/master/tools
4) Each frame is processed to locate the lip area of the face. A cut is made for the frame at the mouth area. As a result, the final frame will only contain the mouth area in order to isolate any interference or noise from the background or other parts of the face. 
5) Each frame is resized to 112X112 pixels in order to reduce the dataset size without affecting the accuracy.
6) Each pixel value in a frame is normalized to a range between 0 and  1 without distorting differences between the frame’s pixels. This is achieved by dividing each pixel value by 255.


# How to test



## Pre-trained Weights
If you want to use pre-trained weights for future workers, you can contact a member of our team to take it .



# Results
<table>
<thead>
  <tr>
    <th>Operation Mode</th>
    <th colspan="1">VO Model</th>
    <th colspan="1">(WER)</th>
    <th colspan="1">(CER)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>Greedy accuracy</td>
  </tr>
  <tr>
    <td>VO</td>
    <td>60.2%</td>
    <td>39.8%</td>
    <td>28.1%</td>
  </tr>
  <tr>
  </tr>
  <tr>
</tbody>
</table>






# License
All rights reserved to our teams who made the model and developed it and built the application


# Contact
You can communicate through one of team by following e-mails to obtain the pre-trained model and the processed dataset:

1- mohamedgasser230@gmail.com

2- 01062416052ss@gmail.com

3- Ma7157563@gmail.com

4- mostafakotb359@gmail.com

5- omarmo5tar12@gmail.com


# References
1- The pre-trained weights of the Visual Frontend and the Model have been obtained from
https://github.com/smeetrs/deep_avsr GitHub repository.

2- We also used this paper in our work https://www.sciencedirect.com/science/article/pii/S1110866522000433

3- We used the LRW, LRS2 and LRS3 lip reading datasets from the BBC https://www.robots.ox.ac.uk/~vgg/data/lip_reading/







