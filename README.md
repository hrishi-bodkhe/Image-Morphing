# Image-Morphing
This was a part of the assignments of the course Digital Image Processing (CS517) taken in IIT Ropar (Jan 2023 - May 2023)

**Reference:** https://github.com/spmallick/learnopencv/tree/master/FaceMorph

## Requirements

Some libraries and modules are used which are needed to be installed first using the following commands:-

A. Installing dilb library:-

    conda install -c conda-forge dlib
    pip install --upgrade dlib
B. Installing the moviepy module to create gif:-

    pip install moviepy
    
**Note:** "shape_predictor_68_face_landmarks.dat" - This file is needed to detect 68 tie points in image using dlib library, which is included in the folder. The same file can be found here:- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
This file is not contained in the repo. After downloading the repo, download the above file, extract it and add it in the same folder.

## Input & Output

Both parts of the assignment are done in the same notebook. Functions are defined in the beginning. A markdown indicates from where PART A & PART B starts.

The folder consists of the images used:- "pic1.jpg" & "pic2.jpg". The final output is stored as "eval1A.gif" & "morphedImageB.gif" in the same folder

Pic 1 Reference:- https://raw.githubusercontent.com/KubricIO/face-morphing/master/demos/ben.jpg

![pic1](https://github.com/hrishi-bodkhe/Image-Morphing/assets/52168214/62530e8d-d597-4f62-86e5-88ca4e9f0ad0)

Pic 2 Reference:- https://raw.githubusercontent.com/KubricIO/face-morphing/master/demos/morgan.jpg

![pic2](https://github.com/hrishi-bodkhe/Image-Morphing/assets/52168214/be5605b9-f30a-4968-9b99-f842389b54d5)


## Detecting Tie Points

![download](https://github.com/hrishi-bodkhe/Image-Morphing/assets/52168214/7aee0b73-94db-4e8e-95d5-4aa06e397c80) ![download](https://github.com/hrishi-bodkhe/Image-Morphing/assets/52168214/83fd12e6-4437-4012-85a1-509a264743d4)

## Triangulation

![download](https://github.com/hrishi-bodkhe/Image-Morphing/assets/52168214/b9fbba97-8636-42ad-a475-22131caea65d)  ![download](https://github.com/hrishi-bodkhe/Image-Morphing/assets/52168214/5a61a270-2623-4878-87cf-e3e06f684b73)

## Morphing

![download](https://github.com/hrishi-bodkhe/Image-Morphing/assets/52168214/5688c7c1-e6ed-48a3-9244-8dc14af9f8ab)

## Final Output (.gif file)

![morphedImageB](https://github.com/hrishi-bodkhe/Image-Morphing/assets/52168214/07058cd9-e245-437d-afb9-c84da2c6e497)
