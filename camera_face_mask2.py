import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

webcam  = cv2.VideoCapture(0)
#read classification
face_cascade = cv2.CascadeClassifier("Detect/haarcascade_frontalface_default.xml")
np.set_printoptions(suppress=True)
# Load model
model = load_model('converted_keras/keras_model.h5')
size = (224, 224)

while (True):
    check , frame = webcam.read() 
    image_org = frame.copy()
    if check == True :
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        #จำแนกใบหน้า
        face_detect = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        #แสดงตำแหน่งที่เจอใบหน้า
        print(f'There are {len(face_detect)} faces found.')
        
        for (x,y,w,h) in face_detect:
            cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])

           
            # Create the array of the right shape to feed into the keras model
            # The 'length' or number of images you can put into the array is
            # determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            # Replace this with the path to your image
            image = cface_rgb
            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
            
            image = ImageOps.fit(image, size, Image.ANTIALIAS)

            #turn the image into a numpy array
            image_array = np.asarray(image)
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = model.predict(data)
            print(prediction)

            if prediction[0][0] > prediction[0][1]:
                cv2.putText(frame, "Masked", (x,y-7), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)
            else:
                cv2.putText(frame, "Non-Masked", (x,y-7), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), thickness=2)


        cv2.imshow("Mask Dectection", frame)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break
    

webcam.release()
cv2.destroyAllWindows()