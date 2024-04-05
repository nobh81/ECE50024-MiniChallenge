import cv2
import os
import shutil

def cropping():
    cnt = 0

    frontalface= r'C:\Users\User\Desktop\50024\Mini_challenge\haarcascade_frontalface_default.xml'
    face_cascade= cv2.CascadeClassifier(frontalface)
    train_small_folder= r'C:\Users\User\Desktop\50024\Mini_challenge\train_small'
    train_large_folder= r'C:\Users\User\Desktop\50024\Mini_challenge\train\train'
    cropped_small= r'C:\Users\User\Desktop\50024\Mini_challenge\cropped_small'
    cropped_large= r'C:\Users\User\Desktop\50024\Mini_challenge\cropped_large'
    final_test= r'C:\Users\User\Desktop\50024\Mini_challenge\test'
    test_crop= r'C:\Users\User\Desktop\50024\Mini_challenge\test_crop3'

    for img_name in os.listdir(final_test):
        img_path= os.path.join(final_test, img_name)
        img= cv2.imread(img_path)
        nxt= int(img_name.split('.')[0])+1
        nxt_idx= f'{nxt}.jpg'

        while img is None:
            print(f"can't not load image {img_path}. Read next")
            img=  cv2.imread(os.path.join(final_test, nxt_idx))
            nxt+=1
            nxt_idx= f'{nxt}.jpg'
            
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray, 1.1, 4)

        if faces is not None and len(faces) > 0:
            largest_face_area= 0
            largest_face= None
            for (x, y, w, h) in faces:#get largest face if there are more than one faces
                area= w * h
                if area > largest_face_area:
                    largest_face_area= area
                    largest_face= (x, y, w, h)
            
            if largest_face is not None:
                x, y, w, h= largest_face
                crop_img= img[y:y+h, x:x+w]
                
                output_filename= f"{os.path.splitext(img_name)[0]}{os.path.splitext(img_name)[1]}"
                cv2.imwrite(os.path.join(test_crop, output_filename), crop_img)

        cnt+= 1
        if cnt%500==0:
            print(f'{cnt} finished')
        

    current_indices= {int(img_name.split('.')[0]) for img_name in os.listdir(test_crop)}
    missing_indices= set(range(0, 4977)) - current_indices

    for idx in sorted(missing_indices):
        source_path= os.path.join(final_test, f'{idx}.jpg')
        destination_path= os.path.join(test_crop, f'{idx}.jpg')
        shutil.copy(source_path, destination_path)
            

cropping()
