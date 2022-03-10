import cv2
import numpy as np
import face_recognition

# def predictImg(input_img,opencv_net):
#     input_img = input_img.astype(np.float32)
#     input_img = cv2.resize(input_img, (256, 256))
#     # define preprocess parameters
#     mean = np.array([0.485, 0.456, 0.406]) * 255.0
#     scale = 1 / 255.0
#     std = [0.229, 0.224, 0.225]
#     # prepare input blob to fit the model input:
#     # 1. subtract mean
#     # 2. scale to set pixel values from 0 to 1
#     input_blob = cv2.dnn.blobFromImage(
#         image=input_img,
#         scalefactor=scale,
#         size=(224, 224),  # img target size
#         mean=mean,
#         swapRB=True,  # BGR -> RGB
#         crop=True  # center crop
#     )
#     # 3. divide by std
#     input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
#
#     # set OpenCV DNN input
#     opencv_net.setInput(input_blob)
#     # OpenCV DNN inference
#     out = opencv_net.forward()
#     #print("OpenCV DNN prediction: \n")
#     #print("* shape: ", out.shape)
#     # get the predicted class ID
#     imagenet_class_id = np.argmax(out)
#     # get confidence
#     confidence = out[0][imagenet_class_id]
#
#     return imagenet_class_id

def img_crop(image):
    #print(path)
    #image = cv2.imread(path)
    face_location = face_recognition.face_locations(image)
    if (len(face_location) > 0):
        top, right, bottom, left = face_location[0]
        padY = int((bottom - top) * .5)
        padX = int((right - left) * .5)
        top = max(0, top - padY)
        bottom = min(image.shape[0], bottom + padY)
        left = max(left, top - padX)
        right = min(image.shape[1], right + padX)
        face_image = image[top:bottom, left:right]
        face_image = cv2.resize(face_image, (224, 224))
        #new_path = path.replace('images', 'images_crop')
        # paths = path.split('/')
        # path2 = "data/original_sequences/youtube/c40/images_crop"+'/'+paths[-1]
        #print(new_path)
        #cv2.imwrite("test1.png", face_image)
        #cv2.imshow("img)
        return face_image
    else:
        print("no face")
        return 0


def predictImg(input_img, opencv_net):
    input_img = img_crop(input_img)
    if not isinstance(input_img, int):
        # copy_img = input_img.copy()
        input_img = input_img.astype(np.float32)
        #input_img = cv2.resize(input_img, (256, 256))
        # define preprocess parameters
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        scale = 1 / 255.0
        std = [0.229, 0.224, 0.225]
        # prepare input blob to fit the model input:
        # 1. subtract mean
        # 2. scale to set pixel values from 0 to 1
        input_blob = cv2.dnn.blobFromImage(
            image=input_img,
            scalefactor=scale,
            size=(224, 224),  # img target size
            mean=mean,
            swapRB=True,  # BGR -> RGB
            crop=True  # center crop
        )
        # 3. divide by std
        input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

        # set OpenCV DNN input
        opencv_net.setInput(input_blob)
        # OpenCV DNN inference
        out = opencv_net.forward()
        #print("OpenCV DNN prediction: \n")
        #print("* shape: ", out.shape)
        # get the predicted class ID
        imagenet_class_id = np.argmax(out)
        # get confidence
        confidence = out[0][imagenet_class_id]
        # if(imagenet_class_id):
        #     cv2.imwrite("saved_test/Real"+str(n)+".png", copy_img)
        # else:
        #     cv2.imwrite("saved_test/Fake" + str(n) + ".png", copy_img)
        # print("predict", imagenet_class_id)
        return imagenet_class_id
    else:
        return -1
def predVideo(video_path, opencv_net, p=.5):
    #opencv_net should be updated according to modelType
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    fcount = 0
    pred = 0
    while success:
        # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
        fcount+=1
        if(fcount%5==0):
            pre = predictImg(image,opencv_net)
            # print('Read a new frame: ', success)
            if(pre!=-1):
                pred+=pre
                count += 1
        success, image = vidcap.read()

    print(pred, count)
    if (count == 0):
        print("Video not found")
        return
    if(pred/count>= p):
        return True
    else:
        return False

# read the image

def ret_prediction(path,type_option,opencv_net):
    if type_option==11:
        try:
            input_img = cv2.imread(path, cv2.IMREAD_COLOR)
            return predictImg(input_img,opencv_net)
        except:
            print('Not Defined')
    else:
        try:
            return predVideo(path,opencv_net)
        except:
            print('Not Defined')


#path = "upload/585.mp4"
#print(ret_prediction(path,22,opencv_net2))
#input_img = cv2.imread(fakePath, cv2.IMREAD_COLOR)
#print("Fake img Pred", predictImg(input_img))
#print(ret_prediction(path))

#fakePath = "upload/585.mp4"
#realPath = "D:/Work/AI_ITI/Project/data/original_sequences/actors/raw/videos/01__kitchen_pan.mp4"

#print("Fake Video Pred", predVideo(fakePath))
#print("Real Video Pred", predVideo(realPath))

"""
#vidcap = cv2.VideoCapture('D:/Work/AI_ITI/Project/data/manipulated_sequences/FaceSwap/raw/videos/183_253.mp4')
vidcap = cv2.VideoCapture('D:/Work/AI_ITI/Project/data/original_sequences/actors/raw/videos/01__kitchen_pan.mp4')
success, image = vidcap.read()
count = 0
pred = 0
while success:
  #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  pred += predictImg(image)
  success, image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1

print(pred, count)
"""
"""
input_img = input_img.astype(np.float32)
input_img = cv2.resize(input_img, (256, 256))
# define preprocess parameters
mean = np.array([0.485, 0.456, 0.406]) * 255.0
scale = 1 / 255.0
std = [0.229, 0.224, 0.225]
# prepare input blob to fit the model input:
# 1. subtract mean
# 2. scale to set pixel values from 0 to 1
input_blob = cv2.dnn.blobFromImage(
    image=input_img,
    scalefactor=scale,
    size=(224, 224),  # img target size
    mean=mean,
    swapRB=True,  # BGR -> RGB
    crop=True  # center crop
)
# 3. divide by std
input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

# set OpenCV DNN input
opencv_net.setInput(input_blob)
# OpenCV DNN inference
out = opencv_net.forward()
print("OpenCV DNN prediction: \n")
print("* shape: ", out.shape)
# get the predicted class ID
imagenet_class_id = np.argmax(out)
# get confidence
confidence = out[0][imagenet_class_id]
print("* class ID: {}, label: ".format(imagenet_class_id))
print("* confidence: {:.4f}".format(confidence))
print(out)
"""