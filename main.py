############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import threading
import shutil
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

############################################# FUNCTIONS ################################################

# Initialize key variable at the start of the file
global key
key = ''

def assure_path_exists(path):
    try:
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    except Exception as e:
        logging.error(f"Error creating directory: {e}")
        messagebox.showerror(title='Error', message=f'Could not create directory: {str(e)}')

##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)

###################################################################################

def contact():
    messagebox.showinfo(title='Contact us', message="Please contact us on : 'xxxxxxxxxxxxx@gmail.com' ")

###################################################################################

def check_haarcascadefile():
    cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    exists = os.path.isfile(cascade_path)
    if exists:
        return cascade_path
    else:
        logging.error("Haar cascade file not found")
        messagebox.showerror(title='Some file missing', message='Please contact us for help')
        window.destroy()
        return None

###################################################################################

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            logging.error("No password entered")
            messagebox.showerror(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            logging.info("New password registered")
            messagebox.showinfo(title='Password Registered', message='New password was registered successfully!!')
            return
    op = (old.get())
    newp= (new.get())
    nnewp = (nnew.get())
    if (op == key):
        if(newp == nnewp):
            txf = open("TrainingImageLabel\psd.txt", "w")
            txf.write(newp)
        else:
            logging.error("Passwords do not match")
            messagebox.showerror(title='Error', message='Confirm new password again!!!')
            return
    else:
        logging.error("Incorrect old password")
        messagebox.showerror(title='Wrong Password', message='Please enter correct old password.')
        return
    logging.info("Password changed successfully")
    messagebox.showinfo(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

###################################################################################

def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False,False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(master,text='    Enter Old Password',bg='white',font=('times', 12, ' bold '))
    lbl4.place(x=10,y=10)
    global old
    old=tk.Entry(master,width=25 ,fg="black",relief='solid',font=('times', 12, ' bold '),show='*')
    old.place(x=180,y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black",relief='solid', font=('times', 12, ' bold '),show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid',font=('times', 12, ' bold '),show='*')
    nnew.place(x=180, y=80)
    cancel=tk.Button(master,text="Cancel", command=master.destroy ,fg="black"  ,bg="red" ,height=1,width=25 , activebackground = "white" ,font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48", height = 1,width=25, activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()

#####################################################################################

def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            logging.error("No password entered")
            messagebox.showerror(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            logging.info("New password registered")
            messagebox.showinfo(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if (password == key):
        TrainImages()
    elif (password == None):
        pass
    else:
        logging.error("Incorrect password")
        messagebox.showerror(title='Wrong Password', message='You have entered wrong password')

######################################################################################

def clear():
    txt.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)

def clear2():
    txt2.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)

#######################################################################################

def TakeImages():
    def capture_images():
        try:
            cascade_path = check_haarcascadefile()
            if cascade_path is None:
                return
                
            columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
            
            # Ensure required directories exist
            assure_path_exists("StudentDetails/")
            assure_path_exists("TrainingImage/")
            
            # Get the next serial number
            serial = 1
            exists = os.path.isfile("StudentDetails\StudentDetails.csv")
            if exists:
                with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
                    reader1 = csv.reader(csvFile1)
                    next(reader1)  # Skip header
                    for _ in reader1:
                        serial += 1
            else:
                with open("StudentDetails\StudentDetails.csv", 'a+', newline='') as csvFile1:
                    writer = csv.writer(csvFile1)
                    writer.writerow(columns)
            
            Id = txt.get().strip()
            name = txt2.get().strip()
            
            if not Id or not name:
                logging.error("ID or name not entered")
                messagebox.showerror(title='Error', message='Please enter both ID and Name!')
                return
                
            try:
                # Validate ID is numeric
                int(Id)
            except:
                logging.error("Invalid ID")
                messagebox.showerror(title='Error', message='ID must be numeric!')
                return
            
            # Check if ID already exists
            if exists:
                with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
                    reader1 = csv.reader(csvFile1)
                    next(reader1)  # Skip header
                    for row in reader1:
                        if len(row) > 2 and row[2].strip() == Id:
                            logging.error("ID already exists")
                            messagebox.showerror(title='Error', message='ID already exists!')
                            return
            
            # Initialize camera
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                # Try alternative camera indices
                for idx in [1, 2, -1]:
                    cam = cv2.VideoCapture(idx)
                    if cam.isOpened():
                        break
                
                if not cam.isOpened():
                    logging.error("Failed to open camera")
                    messagebox.showerror(title='Error', message='Could not access camera! Please check camera connection.')
                    return
            
            detector = cv2.CascadeClassifier(cascade_path)
            if detector.empty():
                logging.error("Failed to load face detector")
                messagebox.showerror(title='Error', message='Failed to load face detector!')
                cam.release()
                return
                
            sampleNum = 0
            
            message1.configure(text='Starting face capture... Please look at the camera')
            window.update_idletasks()
            
            while True:
                ret, img = cam.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
                    
                    # Ensure the face region is large enough
                    if w < 100 or h < 100:
                        cv2.putText(img, "Move Closer", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        continue
                    
                    # Save the captured face
                    face_img = gray[y:y+h, x:x+w]
                    
                    # Ensure image quality
                    if cv2.mean(face_img)[0] < 50:  # Too dark
                        cv2.putText(img, "More Light Needed", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        continue
                    
                    sampleNum += 1
                    # Save with consistent naming
                    img_name = f"{name}.{Id}.{sampleNum}.jpg"
                    img_path = os.path.join("TrainingImage", img_name)
                    
                    # Resize face image to consistent size
                    face_img = cv2.resize(face_img, (200, 200))
                    
                    # Save the image
                    try:
                        cv2.imwrite(img_path, face_img)
                    except Exception as e:
                        logging.error(f"Error saving image {img_path}: {e}")
                        continue
                    
                    cv2.putText(img, f"Images Captured: {sampleNum}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    
                    # Update progress in UI
                    message1.configure(text=f'Captured Image {sampleNum}/100')
                    window.update_idletasks()
                
                cv2.imshow('Capturing Face', img)
                
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                elif sampleNum >= 100:  # Capture 100 samples
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            if sampleNum < 10:  # Ensure we captured enough samples
                logging.error("Not enough face images captured")
                messagebox.showerror(title='Error', message='Not enough face images captured. Please try again!')
                # Clean up any captured images
                for i in range(1, sampleNum + 1):
                    try:
                        os.remove(os.path.join("TrainingImage", f"{name}.{Id}.{i}.jpg"))
                    except:
                        pass
                return
            
            # Save student details
            row = [serial, '', Id, '', name]
            with open('StudentDetails\StudentDetails.csv', 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            
            # Update UI
            message1.configure(text="Images Saved Successfully! Click Train Images.")
            message.configure(text="")
            
            # Refresh profile list and update count
            load_existing_profiles()
            
            # Clear input fields
            txt.delete(0, 'end')
            txt2.delete(0, 'end')
            
        except Exception as e:
            logging.error(f"Error in capture_images: {e}")
            if 'cam' in locals():
                cam.release()
            cv2.destroyAllWindows()
            messagebox.showerror(title='Error', message=f'An error occurred: {e}')
            return
    
    # Start image capture in a separate thread
    capture_thread = threading.Thread(target=capture_images)
    capture_thread.daemon = True
    capture_thread.start()

########################################################################################

def getImagesAndLabels(path):
    try:
        # Get all file paths
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        Ids = []
        
        # Initialize face detector with absolute path
        cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            logging.error("Failed to load face detector")
            raise ValueError("Failed to load face detector. Check haarcascade file.")
        
        logging.debug(f"Processing {len(imagePaths)} images...")
        
        for imagePath in imagePaths:
            try:
                # Read image and convert to grayscale
                logging.debug(f"Processing image: {imagePath}")
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                
                # Get the label (ID) from the image path
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                
                # Detect faces with more lenient parameters
                faces_rect = detector.detectMultiScale(
                    imageNp,
                    scaleFactor=1.2,  # More lenient scale factor
                    minNeighbors=3,   # Reduced for better detection
                    minSize=(20, 20)  # Smaller minimum face size
                )
                
                if len(faces_rect) > 0:
                    x, y, w, h = faces_rect[0]  # Take the first face
                    
                    # Add padding to the face region
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(imageNp.shape[1] - x, w + 2*padding)
                    h = min(imageNp.shape[0] - y, h + 2*padding)
                    
                    # Extract and process face region
                    face_img = imageNp[y:y+h, x:x+w]
                    
                    # Enhance image contrast
                    face_img = cv2.equalizeHist(face_img)
                    
                    # Resize to consistent size
                    face_img = cv2.resize(face_img, (200, 200))
                    
                    faces.append(face_img)
                    Ids.append(Id)
                    logging.debug(f"Added face with ID: {Id}")
                else:
                    logging.debug(f"No face detected in {imagePath}")
                    
            except Exception as e:
                logging.error(f"Error processing {imagePath}: {e}")
                continue
                
        logging.debug(f"Successfully processed {len(faces)} faces")
        
        if len(faces) == 0:
            logging.error("No valid faces found")
            raise ValueError("No valid faces found in training images. Please ensure images contain clear, well-lit faces.")
            
        return faces, Ids
        
    except Exception as e:
        logging.error(f"Error in getImagesAndLabels: {e}")
        raise

def TrainImages():
    try:
        logging.info("Starting training process...")
        
        # Check if training directory exists and has files
        if not os.path.exists("TrainingImage"):
            logging.error("Training image directory not found")
            messagebox.showerror(title='No Data', message='Please add images first!')
            return False
            
        if len(os.listdir("TrainingImage")) == 0:
            logging.error("No images in training directory")
            messagebox.showerror(title='No Data', message='Please add images first!')
            return False
            
        # Ensure TrainingImageLabel directory exists
        assure_path_exists("TrainingImageLabel/")
        
        # Initialize recognizer
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            logging.debug("Created face recognizer")
        except Exception as e:
            logging.error(f"Error creating recognizer: {e}")
            messagebox.showerror(title='Error', message='Failed to initialize face recognizer. Please reinstall opencv-contrib-python')
            return False
        
        # Get faces and IDs
        try:
            logging.debug("Getting images and labels...")
            faces, Ids = getImagesAndLabels("TrainingImage")
            logging.debug(f"Got {len(faces)} faces and {len(Ids)} IDs")
            
            if len(faces) == 0:
                logging.error("No faces found in training images")
                messagebox.showerror(title='Error', message='No faces found in training images!')
                return False
            
            # Convert lists to numpy arrays
            faces = np.array(faces)
            Ids = np.array(Ids)
            
            logging.debug("Training model...")
            # Train the model
            recognizer.train(faces, Ids)
            logging.info("Model trained successfully")
            
            # Save the model
            model_path = os.path.join("TrainingImageLabel", "Trainner.yml")
            recognizer.save(model_path)
            logging.debug(f"Model saved to {model_path}")
            
            res = "Image Trained Successfully!"
            message1.configure(text=res)
            
            return True
            
        except ValueError as ve:
            logging.error(f"ValueError in training: {ve}")
            messagebox.showerror(title='Error', message=str(ve))
            return False
        except Exception as e:
            logging.error(f"Error in training: {e}")
            messagebox.showerror(title='Error', message=f'Training failed: {e}')
            return False
            
    except Exception as e:
        logging.error(f"Error in TrainImages: {e}")
        messagebox.showerror(title='Error', message=str(e))
        return False

############################################################################################3

def TrackImages():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Check for trained model
        if not os.path.exists("TrainingImageLabel/Trainner.yml"):
            logging.error("Trained model not found")
            messagebox.showerror(title='No Data', message='Please train the model first!')
            return
            
        # Load the trained model
        recognizer.read("TrainingImageLabel/Trainner.yml")
        
        # Load face cascade
        cascade_path = check_haarcascadefile()
        if cascade_path is None:
            return
            
        faceCascade = cv2.CascadeClassifier(cascade_path)
        
        # Load student details into a dictionary for quick lookup
        student_dict = {}
        if os.path.exists("StudentDetails/StudentDetails.csv"):
            df = pd.read_csv("StudentDetails/StudentDetails.csv", usecols=['ID', 'NAME'])
            for _, row in df.iterrows():
                student_dict[str(row['ID'])] = row['NAME']
        
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        while True:
            ret, im = cam.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            
            for(x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                
                if conf < 50:  # Lower threshold for more accurate predictions
                    if str(Id) in student_dict:
                        tt = student_dict[str(Id)]
                        cv2.putText(im, f"{tt} ({conf:.0f}%)", (x+5, y-5), font, 1, (255, 255, 255), 2)
                    
                        # Mark attendance
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        
                        # Create attendance file if it doesn't exist
                        attendance_file = f"Attendance/Attendance_{date}.csv"
                        if not os.path.exists(attendance_file):
                            with open(attendance_file, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(['Id', 'Name', 'Time'])
                        
                        # Check if already marked attendance
                        marked = False
                        with open(attendance_file, 'r') as f:
                            reader = csv.reader(f)
                            next(reader)  # Skip header
                            for row in reader:
                                if row[0] == str(Id):
                                    marked = True
                                    break
                        
                        # Mark attendance if not already marked
                        if not marked:
                            with open(attendance_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([Id, tt, timeStamp])
                else:
                    cv2.putText(im, 'Unknown', (x+5, y-5), font, 1, (0, 0, 255), 2)
            
            cv2.imshow('Taking Attendance', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
        # Debug print statements
        logging.debug("Attempting to open attendance file")
        logging.debug(f"Current working directory: {os.getcwd()}")
        
        # Open the most recently created attendance file
        attendance_dir = "Attendance"
        logging.debug(f"Attendance directory: {attendance_dir}")
        logging.debug(f"Directory exists: {os.path.exists(attendance_dir)}")
        
        # Get the list of files
        files = os.listdir(attendance_dir)
        logging.debug(f"Files in directory: {files}")
        
        # Filter attendance files
        attendance_files = [f for f in files if f.startswith("Attendance_") and f.endswith(".csv")]
        logging.debug(f"Attendance files: {attendance_files}")
        
        if attendance_files:
            # Get the most recently created attendance file
            latest_attendance_file = max(
                [os.path.join(attendance_dir, f) for f in attendance_files], 
                key=os.path.getctime
            )
            logging.debug(f"Latest attendance file: {latest_attendance_file}")
            
            # Open the file with the default application
            try:
                os.startfile(latest_attendance_file)
                logging.debug(f"Successfully opened: {latest_attendance_file}")
            except Exception as e:
                logging.error(f"Error opening file: {e}")
                messagebox.showerror("Error", f"Could not open attendance file: {e}")
    
    except Exception as e:
        logging.error(f"Error in TrackImages: {e}")
        messagebox.showerror(title='Error', message=str(e))

######################################## USED STUFFS ############################################
    
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")

mont={'01':'January',
      '02':'February',
      '03':'March',
      '04':'April',
      '05':'May',
      '06':'June',
      '07':'July',
      '08':'August',
      '09':'September',
      '10':'October',
      '11':'November',
      '12':'December'
      }

######################################## GUI FRONT-END ###########################################

PRIMARY_COLOR = "#2c3e50"  # Dark blue-gray
SECONDARY_COLOR = "#3498db"  # Bright blue
ACCENT_COLOR = "#e74c3c"  # Red
BG_COLOR = "#ecf0f1"  # Light gray
TEXT_COLOR = "#2c3e50"  # Dark blue-gray

try:
    logging.info("Starting Face Recognition Attendance System")
    
    window = tk.Tk()
    window.title("Face Recognition Attendance System")
    
    logging.debug("Configuring window properties")
    window.geometry('1280x720')  # Adjust size as needed
    window.resizable(True, True)
    
    window.configure(background=BG_COLOR)

    message3 = tk.Label(window, text="Face Recognition Attendance System", fg=PRIMARY_COLOR, bg=BG_COLOR, width=55, height=1, font=('Helvetica', 29, 'bold'))
    message3.place(relx=0.5, rely=0.05, anchor="center")

    # Create a separator line
    separator = tk.Frame(window, height=2, bg=SECONDARY_COLOR)
    separator.place(relx=0.15, rely=0.13, relwidth=0.7)

    # Create a frame for date and time with a subtle background
    datetime_frame = tk.Frame(window, bg=BG_COLOR)
    datetime_frame.place(relx=0.5, rely=0.16, anchor="center")

    # Create a decorative border around date and time
    datetime_border = tk.Frame(datetime_frame, bg=SECONDARY_COLOR, padx=2, pady=2)
    datetime_border.pack(padx=5, pady=5)

    inner_frame = tk.Frame(datetime_border, bg=BG_COLOR, padx=15, pady=8)
    inner_frame.pack()

    date_label = tk.Label(inner_frame, text="Date: " + datetime.datetime.now().strftime("%B %d, %Y"), 
                         bg=BG_COLOR, fg=PRIMARY_COLOR, 
                         font=('Helvetica', 12, 'bold'))
    date_label.pack(side=tk.LEFT, padx=20)

    # Add a vertical separator between date and time
    tk.Frame(inner_frame, width=2, bg=SECONDARY_COLOR).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

    clock = tk.Label(inner_frame, bg=BG_COLOR, fg=PRIMARY_COLOR, font=('Helvetica', 12, 'bold'))
    clock.pack(side=tk.LEFT, padx=20)

    # Create frames for better organization
    frame1 = tk.Frame(window, bg=BG_COLOR)
    frame1.place(relx=0.11, rely=0.25, relwidth=0.39, relheight=0.72)

    frame2 = tk.Frame(window, bg=BG_COLOR)
    frame2.place(relx=0.51, rely=0.25, relwidth=0.38, relheight=0.72)

    fr_head1 = tk.Label(frame1, text="Already Registered Students", fg="white", bg=SECONDARY_COLOR, font=('Helvetica', 16, 'bold'))
    fr_head1.place(x=0, y=0, relwidth=1)

    fr_head2 = tk.Label(frame2, text="Register New Student", fg="white", bg=SECONDARY_COLOR, font=('Helvetica', 16, 'bold'))
    fr_head2.place(x=0, y=0, relwidth=1)

    # Setup treeview
    tv = ttk.Treeview(frame1, height=13, columns=('name', 'id', 'serial'))
    tv.column('#0', width=0, stretch=tk.NO)  # Hidden first column
    tv.column('name', width=150)
    tv.column('id', width=100)
    tv.column('serial', width=100)

    tv.heading('#0', text='')
    tv.heading('name', text='Name')
    tv.heading('id', text='ID')
    tv.heading('serial', text='Serial No.')

    tv.place(x=0, y=40, relwidth=0.9, height=300)

    scrollbar = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
    scrollbar.place(relx=0.9, y=40, relwidth=0.1, height=300)

    tv.configure(yscrollcommand=scrollbar.set)

    # Add Mark Attendance button below treeview
    trackImg = tk.Button(frame1, text="Mark Attendance", command=TrackImages, 
                        fg="white", bg=SECONDARY_COLOR,
                        width=30, height=1, 
                        activebackground=SECONDARY_COLOR, 
                        activeforeground="white",
                        font=('Helvetica', 14))
    trackImg.place(x=10, y=380)

    lbl = tk.Label(frame2, text="Enter ID", width=15, height=1, fg=TEXT_COLOR, bg=BG_COLOR, font=('Helvetica', 12))
    lbl.place(x=30, y=60)
    txt = tk.Entry(frame2, width=25, fg=TEXT_COLOR, font=('Helvetica', 12))
    txt.place(x=30, y=90)

    lbl2 = tk.Label(frame2, text="Enter Name", width=15, fg=TEXT_COLOR, bg=BG_COLOR, font=('Helvetica', 12))
    lbl2.place(x=30, y=130)
    txt2 = tk.Entry(frame2, width=25, fg=TEXT_COLOR, font=('Helvetica', 12))
    txt2.place(x=30, y=160)

    message = tk.Label(frame2, text="", bg=BG_COLOR, fg=TEXT_COLOR, width=39, height=1, font=('Helvetica', 12))
    message.place(x=7, y=450)

    message1 = tk.Label(frame2, text="", bg=BG_COLOR, fg=TEXT_COLOR, width=39, height=1, font=('Helvetica', 12))
    message1.place(x=7, y=480)

    clearButton = tk.Button(frame2, text="Clear", command=clear, fg="white", bg=ACCENT_COLOR, width=10,
                            activebackground=ACCENT_COLOR, activeforeground="white", font=('Helvetica', 12))
    clearButton.place(x=335, y=90)

    clearButton2 = tk.Button(frame2, text="Clear", command=clear2, fg="white", bg=ACCENT_COLOR, width=10,
                            activebackground=ACCENT_COLOR, activeforeground="white", font=('Helvetica', 12))
    clearButton2.place(x=335, y=160)

    takeImg = tk.Button(frame2, text="Capture Images", command=TakeImages, 
                       fg="white", bg=SECONDARY_COLOR,
                       width=30, height=1, 
                       activebackground=SECONDARY_COLOR, 
                       activeforeground="white",
                       font=('Helvetica', 14))
    takeImg.place(x=30, y=260)  

    trainImg = tk.Button(frame2, text="Train Images", 
                        command=TrainImages, 
                        fg="white", 
                        bg=SECONDARY_COLOR,
                        width=30, height=1, 
                        activebackground=SECONDARY_COLOR, 
                        activeforeground="white",
                        font=('Helvetica', 14))
    trainImg.place(x=30, y=320)  

    clearButton = tk.Button(
        frame2,
        text="Clear All Profiles",
        command=lambda: clear_all_profiles(),
        fg="white",
        bg="#ff4444",
        width=30,  
        height=1,  
        activebackground="#ff6666",
        activeforeground="white",
        font=('Helvetica', 14)  
    )
    clearButton.place(x=30, y=380)  

    def remove_profile():
        try:
            # Ask for profile ID to remove
            remove_id = tsd.askstring('Remove Profile', 'Enter the ID of the profile to remove:')
            
            if not remove_id:
                return  # User cancelled
            
            # Ensure StudentDetails directory and CSV exist
            csv_path = "StudentDetails/StudentDetails.csv"
            if not os.path.exists(csv_path):
                logging.error("Student details CSV not found")
                messagebox.showerror(title="Error", message="No profiles exist!")
                return
            
            # Read existing CSV
            rows = []
            profile_found = False
            with open(csv_path, 'r') as csvFile:
                reader = csv.reader(csvFile)
                headers = next(reader)  # Keep headers
                rows.append(headers)
                
                for row in reader:
                    if row and len(row) > 2 and row[2].strip() == remove_id:
                        profile_found = True
                        # Remove associated training images
                        training_img_path = os.path.join("TrainingImage", f"{remove_id}.jpg")
                        if os.path.exists(training_img_path):
                            os.remove(training_img_path)
                    else:
                        rows.append(row)
            
            # Write updated CSV
            with open(csv_path, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(rows)
            
            # Reload profiles and update treeview
            load_existing_profiles()
            
            # Show appropriate message
            if profile_found:
                message1.configure(text=f'Profile with ID {remove_id} removed successfully!')
                messagebox.showinfo(title="Success", message=f"Profile with ID {remove_id} has been removed.")
            else:
                message1.configure(text=f'No profile found with ID {remove_id}')
                messagebox.showwarning(title="Not Found", message=f"No profile exists with ID {remove_id}.")
    
        except Exception as e:
            logging.error(f"Error in remove_profile: {e}")
            messagebox.showerror(
                title="Error",
                message=f"Failed to remove profile:\n{e}",
                icon='error'
            )

    removeButton = tk.Button(
        frame2,
        text="Remove Profile",
        command=remove_profile,
        fg="white",
        bg="#ff4444",
        width=30,  
        height=1,  
        activebackground="#ff6666",
        activeforeground="white",
        font=('Helvetica', 14)  
    )
    removeButton.place(x=30, y=420)  

    def update_registration_count():
        try:
            if os.path.exists("StudentDetails/StudentDetails.csv"):
                with open("StudentDetails/StudentDetails.csv", 'r') as csvFile:
                    reader = csv.reader(csvFile)
                    next(reader)  # Skip header row
                    count = 0
                    for row in reader:
                        if row and len(row) > 2 and row[2].strip():  # Check for valid ID
                            count += 1
                    logging.debug(f"Current registration count: {count}")  # Just for debugging
        except Exception as e:
            logging.error(f"Error updating count: {e}")

    def load_existing_profiles():
        try:
            # Clear existing items in treeview
            for item in tv.get_children():
                tv.delete(item)
            
            # Ensure StudentDetails directory exists
            if not os.path.exists("StudentDetails"):
                os.makedirs("StudentDetails")
            
            # Create CSV file with headers if it doesn't exist
            csv_path = "StudentDetails/StudentDetails.csv"
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])
            
            # Load existing profiles
            with open(csv_path, 'r') as csvFile:
                reader = csv.reader(csvFile)
                next(reader)  # Skip header
                for row in reader:
                    if row and len(row) > 2 and row[2].strip():  # Check for valid ID
                        serial = row[0]
                        student_id = row[2]
                        name = row[4] if len(row) > 4 else ""
                        # Insert into treeview
                        tv.insert('', 'end', values=(name, student_id, serial))
                    
            logging.debug("Successfully loaded existing profiles")  # Debug print
        
        except Exception as e:
            logging.error(f"Error loading existing profiles: {e}")
            messagebox.showerror(
                title="Error",
                message=f"Failed to load existing profiles:\n{e}",
                icon='error'
            )

    def clear_all_profiles():
        try:
            # Show confirmation dialog
            confirm = messagebox.askyesno(
                title="Confirm Delete",
                message="Are you sure you want to delete all profiles?\nThis action cannot be undone.",
                icon='warning'
            )
            
            if not confirm:
                return
            
            # Clear Training Images
            training_dir = "TrainingImage"
            if os.path.exists(training_dir):
                files = os.listdir(training_dir)
                for file in files:
                    file_path = os.path.join(training_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    
            # Clear Training Labels
            label_dir = "TrainingImageLabel"
            if os.path.exists(label_dir):
                files = os.listdir(label_dir)
                for file in files:
                    file_path = os.path.join(label_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    
            # Clear Attendance Records
            attendance_dir = "Attendance"
            if os.path.exists(attendance_dir):
                files = os.listdir(attendance_dir)
                for file in files:
                    if file.startswith("Attendance_"):
                        file_path = os.path.join(attendance_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        
            # Reset Student Details CSV
            csv_path = "StudentDetails/StudentDetails.csv"
            with open(csv_path, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])
            
            # Clear input fields
            txt.delete(0, 'end')
            txt2.delete(0, 'end')
            
            # Clear treeview
            for item in tv.get_children():
                tv.delete(item)
            
            message1.configure(text='All profiles cleared!')
            
            messagebox.showinfo(
                title="Success",
                message="All profiles have been cleared successfully!",
                icon='info'
            )
        
        except Exception as e:
            logging.error(f"Error in clear_all_profiles: {e}")
            messagebox.showerror(
                title="Error",
                message=f"Failed to clear profiles:\n{e}",
                icon='error'
            )

    # Initial count update and load profiles
    load_existing_profiles()

    # Periodic count update
    def periodic_count_update():
        try:
            update_registration_count()
            window.after(5000, periodic_count_update)  # Update every 5 seconds
        except Exception as update_error:
            logging.error(f"Error in periodic count update: {update_error}")
    
    window.after(1000, periodic_count_update)

    quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, 
                          fg="white", bg=ACCENT_COLOR,
                          width=15, height=1, 
                          activebackground=ACCENT_COLOR, 
                          activeforeground="white",
                          font=('Helvetica', 14),
                          relief="flat",
                          cursor="hand2")
    quitWindow.place(relx=0.5, rely=0.9, anchor="center")

    menubar = tk.Menu(window, relief='ridge')
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label='Change Password', command=change_pass)
    filemenu.add_command(label='Contact Us', command=contact)
    filemenu.add_command(label='Exit', command=window.destroy)
    menubar.add_cascade(label='Help', menu=filemenu)

    window.configure(menu=menubar)

    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            # If the default path doesn't work, try local file
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                raise Exception("Error loading face cascade classifier")
    except Exception as e:
        logging.error(f"Error initializing face detector: {e}")
        messagebox.showerror(title='Error', message='Could not initialize face detection. Please check OpenCV installation.')

    tick()
    # Load existing profiles when application starts
    load_existing_profiles()

    logging.info("Starting Tkinter main loop")
    window.mainloop()

except Exception as e:
    logging.critical(f"Unhandled exception: {e}", exc_info=True)
    messagebox.showerror("Critical Error", f"An unexpected error occurred:\n{e}\n\nPlease check the log file for details.")
