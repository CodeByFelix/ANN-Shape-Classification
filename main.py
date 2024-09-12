from ModelPredict import ModelPredict
import cv2
import PySimpleGUI as sg


predictImage = ModelPredict()
path = ''
sg.theme('LightBlue1')
row1 = [
        [sg.Push(), sg.Image(key='_image'), sg.Push()]
        
        ]
row2 = [
        [sg.Push(), sg.Button('Chose\nImage', key="_fileBrowse", size=(12,2),font=('Helvetica', 12, 'bold')), 
         sg.Push(), 
         sg.Button('Predict', key='_predict', size=(12,2),font=('Helvetica', 12, 'bold')), sg.Push()]
        
        ]

layout = [
        [row1[0]],
        [sg.VPush()],
        [sg.Push(), sg.Text("", key='_text', font=('Helvetica', 12, 'bold')), sg.Push()],
        [row2[0]]
    ]


window = sg.Window("Shape Prediction", layout, size=(600,400), resizable=False, finalize=True)


while True:
    event, value = window.read(timeout=20)
    if event == sg.WIN_CLOSED:
        break
    
    if event == '_fileBrowse':
        path = sg.popup_get_file("Select an Image", no_window=True, file_types=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if path:
            img = cv2.imread(path)
            imgBytes = cv2.imencode('.png', img)[1].tobytes()
            window['_image'].update(data=imgBytes)
            window["_text"].update("")
            
            
    if event == "_predict":
        shape, conf = predictImage.predict(img)
        conf = int (conf * 100)
        window["_text"].update(f"The shape {shape} was predicted with a confidence of {conf}%")


window.close()



