import PySimpleGUI as sg
import numpy as np
import pandas as pd
import sqlite3
import cv2
from keras.models import model_from_json
from sklearn import preprocessing, model_selection, neighbors


# Add some color to the window
sg.theme('DarkTeal9')

data=sqlite3.connect("stress.db")
cur=data.cursor()

cur.execute("create table if not exists patients(Name text,DOB text,Job text,upset integer,control integer,nerv integer,conf integer,way integer,cope integer,irr integer,top integer,anger integer,diff integer, sr real,rr real,t real,lm real,bo real,rem real,sleep real,emo integer,hr real,thoughts text,sl integer)")
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())

#<----------------------------------Training data-------------------------------------------->
df=pd.read_csv('adv_stress.csv')
print(df.head())
print(df.describe())

X= np.array(df.drop(['sl'],axis=1))
Y= np.array(df['sl'])

X_train, X_test, Y_train, Y_test =model_selection.train_test_split(X,Y,test_size=0.2)
clf= neighbors.KNeighborsClassifier(n_neighbors=100)
clf.fit(X_train,Y_train)
accuracy =clf.score(X_test, Y_test)
print(accuracy)




#<-----------------------------Creating instances of test----------------------------------->
state=1
detail = [
    [sg.Text('Please fill out the following fields:',justification='c',expand_x=True)],
    [sg.Text('Full Name', size=(15,1)), sg.InputText(key='Name')],
    [sg.Text('Date of Birth', size=(15,1)), sg.InputText(key='dob')],
    [sg.Text('Job', size=(15,1)), sg.InputText(key='job')],
        [sg.T(' '*40), sg.Button('Next'), sg.Exit()],
    [sg.Push(),sg.Sizegrip()]
]

ques1 =[[sg.Text('Answer these as you see fit:',justification='c',expand_x=True)],
    [sg.Text('In the last month, how often have you been upset because of something that happened unexpectedly?')],
    [sg.Radio('Never', 1),sg.Radio('Almost Never', 1),sg.Radio('Sometimes', 1),sg.Radio('Fairly Often', 1),sg.Radio('Very Often', 1)],
    [sg.Text('In the last month, how often have you felt that you were unable to control the important things in your life?')],
    [sg.Radio('Never', 2),sg.Radio('Almost Never', 2),sg.Radio('Sometimes', 2),sg.Radio('Fairly Often', 2),sg.Radio('Very Often', 2)],
    [sg.Text('In the last month, how often have you felt nervous and stressed?')],
    [sg.Radio('Never', 3),sg.Radio('Almost Never', 3),sg.Radio('Sometimes', 3),sg.Radio('Fairly Often', 3),sg.Radio('Very Often', 3)],
    [sg.Text(' In the last month, how often have you felt confident about your ability to handle your personal problems?')],
    [sg.Radio('Never', 4),sg.Radio('Almost Never', 4),sg.Radio('Sometimes', 4),sg.Radio('Fairly Often', 4),sg.Radio('Very Often', 4)],
    [sg.Text('In the last month, how often have you felt that things were going your way?')],
    [sg.Radio('Never', 5),sg.Radio('Almost Never', 5),sg.Radio('Sometimes', 5),sg.Radio('Fairly Often', 5),sg.Radio('Very Often', 5)],
    [sg.Text('In the last month, how often have you found that you could not cope with all the things that you had to do?')],
    [sg.Radio('Never', 6),sg.Radio('Almost Never', 6),sg.Radio('Sometimes', 6),sg.Radio('Fairly Often', 6),sg.Radio('Very Often', 6)],
    [sg.Text('In the last month, how often have you been able to control irritations in your life?')],
    [sg.Radio('Never', 7),sg.Radio('Almost Never', 7),sg.Radio('Sometimes', 7),sg.Radio('Fairly Often', 7),sg.Radio('Very Often', 7)],
    [sg.Text('In the last month, how often have you felt that you were on top of things?')],
    [sg.Radio('Never', 8),sg.Radio('Almost Never', 8),sg.Radio('Sometimes', 8),sg.Radio('Fairly Often', 8),sg.Radio('Very Often', 8)],
    [sg.Text('. In the last month, how often have you been angered because of things that happened that were outside of your control?')],
    [sg.Radio('Never', 9),sg.Radio('Almost Never', 9),sg.Radio('Sometimes', 9),sg.Radio('Fairly Often', 9),sg.Radio('Very Often', 9)],
    [sg.Text('. In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?')],
    [sg.Radio('Never', 10),sg.Radio('Almost Never', 10),sg.Radio('Sometimes', 10),sg.Radio('Fairly Often', 10),sg.Radio('Very Often', 10)],
    [sg.T(' '*60),sg.Button('Previous',key='p_q1'), sg.Button('Next',key='n_q1'), sg.Exit()]
    ]

ques2=[[sg.Text('Physiological Data',justification='c',expand_x=True)],
    [sg.Text('Snoring Range', size=(15,1)), sg.InputText(key='sr'),
     sg.Text('Respiration Rate', size=(15,1)), sg.InputText(key='rr')],
    [sg.Text('Body Temperature', size=(15,1)), sg.InputText(key='t'),
     sg.Text('Limb Movement Rate', size=(15,1)), sg.InputText(key='lm')],
    [sg.Text('Blood Oxygen Level', size=(15,1)), sg.InputText(key='bo'),
     sg.Text('Eye Movement', size=(15,1)), sg.InputText(key='rem')],
    [sg.Text('Hours of Sleep', size=(15,1)), sg.InputText(key='sr.1'),
     sg.Text('Heart Rate', size=(15,1)), sg.InputText(key='hr')],
    [sg.T(' '*100),sg.Button('Previous',key='p_q2'), sg.Button('Next',key='n_q2'), sg.Exit()]
       ]

thots=[ [sg.Text('Deeper look in you',justification='c',expand_x=True)],
    [sg.Multiline(default_text="Write few words about your thoughts",size=(150,10),key='th')],
        [sg.T(' '*100),sg.Button('Previous',key='p_th'), sg.Button('Next',key='n_th'), sg.Exit()]
        ]
cam=[[sg.Text('Provide visuals:',justification='c',expand_x=True)],
     [sg.InputText(key='file'),sg.FileBrowse()],
     [sg.Text('OR...',justification='c',expand_x=True)],
     [sg.T(' '*30),sg.Button('Open Camera')],
     [sg.T(' '*20),sg.Button('Previous',key='p_cam'), sg.Submit(), sg.Exit()]
     ]
     

layout=[[sg.Col(detail,key='col1'),sg.Col(ques1,visible=False,key='col2'),sg.Col(ques2,visible=False,key='col3'),sg.Col(thots,visible=False,key='col4'),sg.Col(cam,visible=False,key='col5')]]

window = sg.Window('RELIEF', layout)

#<----------------------------------------------functions---------------------------------------------------------------------------->

play=[]
emo=4

def feed():
    cap=cv2.VideoCapture(0)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # load json and create model
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("model/emotion_model.h5")
    print("Loaded model from disk")

    # start the webcam feed
    #cap = cv2.VideoCapture(0)

    # pass here your video path
    # you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
   # cap = cv2.VideoCapture(event[file])
    maxindex=0

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    emo=maxindex
    cap.release()
    cv2.destroyAllWindows()


def out():
    ex=np.array([play])
    print(len(ex))
    ex=ex.reshape(len(ex),-1)
    pred=clf.predict(ex)
    sg.Print("Accuracy of model with 21 Near Neighbors:",accuracy)
    sg.Print(pred,"with confidence of",clf.predict_proba(ex))




    
def clear_input():
    for key in values:
        window[key]('')
    return None

def dump(ev):
    for i in range(50):
        if ev[i]==True and i<5:
            play.append(i)
        if ev[i]== True and i>=5 and i<10:
            play.append(i-5)
        if ev[i]== True and i>=10 and i<15:
            play.append(i-10)
        if ev[i]== True and i>=15 and i<20:
            play.append(i-15)
        if ev[i]== True and i>=20 and i<25:
            play.append(i-20)
        if ev[i]== True and i>=25 and i<30:
            play.append(i-25)
        if ev[i]== True and i>=30 and i<35:
            play.append(i-30)
        if ev[i]== True and i>=35 and i<40:
            play.append(i-35)
        if ev[i]== True and i>=40 and i<45:
            play.append(i-40)
        if ev[i]== True and i>=45 and i<50:
            play.append(i-45)
    play.append(float(ev['sr']))
    play.append(float(ev['rr']))
    play.append(float(ev['t']))
    play.append(float(ev['lm']))
    play.append(float(ev['bo']))
    play.append(float(ev['rem']))
    play.append(float(ev['sr.1']))
    play.append(int(emo))
    play.append(float(ev['hr']))

    print(play)



def col_get():
    print(state)
    if state==1:
        window['col1'].update(visible=True)
        window['col2'].update(visible=False)
        window['col3'].update(visible=False)
        window['col4'].update(visible=False)
        window['col5'].update(visible=False)
    if state==2:
        window['col1'].update(visible=False)
        window['col2'].update(visible=True)
        window['col3'].update(visible=False)
        window['col4'].update(visible=False)
        window['col5'].update(visible=False)
    if state==3:
        window['col1'].update(visible=False)
        window['col2'].update(visible=False)
        window['col3'].update(visible=True)
        window['col4'].update(visible=False)
        window['col5'].update(visible=False)
    if state==4:
        window['col1'].update(visible=False)
        window['col2'].update(visible=False)
        window['col3'].update(visible=False)
        window['col4'].update(visible=True)
        window['col5'].update(visible=False)
    if state==5:
        window['col1'].update(visible=False)
        window['col2'].update(visible=False)
        window['col3'].update(visible=False)
        window['col4'].update(visible=False)
        window['col5'].update(visible=True)
    

while True:
    event, values = window.read() 
    print(event)
    if event == 'Next' or event == 'n_q1' or event == 'n_q2' or event== 'n_th':
        if state>=5:
            state=5
        else:
            state+=1
        col_get()
    if event == 'p_q1' or event == 'p_q2' or event == 'p_th' or event=='p_cam':
        if state<=1:
            state=1
        else:
            state-=1
        col_get()
    if event == 'Open Camera':
        feed()
    if event == sg.WIN_CLOSED or event == 'Exit' or event == 'Exit3':
        break
    if event == 'Clear':
        clear_input()
    if event == 'Submit':
        dump(values)
        out()
        print(values)
##        val=values.values()
##        m=tuple(val)
##        cur.execute("insert into patients values(?,?,?,?)",m)
##        data.commit()
##        sg.popup('Data saved!')
        clear_input()
    if event == 'print':
        for i in cur.execute("select * from patients"):
            print(i)
            
window.close()
data.close()
