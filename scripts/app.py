import os, time, logging, tempfile
import cv2 as cv
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import pymongo
import time
import pandas as pd
import pymongo
import schedule
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from datetime import datetime
from datetime import datetime, timedelta
import pytz


MODEL_DIR = "./runs/detect/train/weights/best.pt"


logging.basicConfig(
    filename="./logs/log.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)


def send_mail(subject, body):
    """
    Send an email notification if food is not detected and log the email in a file.

    Args:
        detail (str): Details of the cage where food is not detected.
    """
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "chiranjeevikumarbattula@gmail.com"  # Chiranjeevi
    password = "qymt kxdm sidv jfet"
    receiver_email = [
        # "krishna@eigenmaps.ai",
        # "adityareddy@eigenmaps.ai",
        # # "zuber@eigenmaps.ai",
        # "zuber.abdul@eizen.ai",
        "chiranjeevikumarbattula@gmail.com",
        # "Janani@lifesciencetrust.com",
        # "bharath@lifesciencetrust.com",
    ]

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(receiver_email)
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)

        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)

        # Log the email in a file
        with open("email_log.log", "a") as log_file:
            log_file.write(
                f"Email sent at {datetime.now()}, "
                f"From: {sender_email}, To: {', '.join(receiver_email)}, Body: {body}\n"
            )
        print(
            f"Email sent at {datetime.now()}, From: {sender_email}, To: {', '.join(receiver_email)}, Body: {body}"
        )
    except Exception as e:
        # Log the error in a file
        with open("email_log.log", "a") as log_file:
            log_file.write(f"Error at {datetime.now()}: {str(e)}\n")

        print(f"An error occurred: {str(e)}")
    finally:
        server.quit()


def main():
    # load a model
    global model
    model = YOLO(MODEL_DIR)

    st.sidebar.header("**Animal Classes**")

    class_names = [
        "Buffalo",
        "Elephant",
        "Rhino",
        "Zebra",
        "Cheetah",
        "Fox",
        "Jaguar",
        "Tiger",
        "Lion",
        "Panda",
    ]

    for animal in class_names:
        st.sidebar.markdown(f"- *{animal.capitalize()}*")

    st.title("Wild Animal Detection & Alerting System")
    # st.write("The aim of this project is to develop an efficient computer vision model capable of real-time wildlife detection.")

    # Load image or video
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "mp4"]
    )

    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            inference_images(uploaded_file)

        if uploaded_file.type.startswith("video"):
            inference_video(uploaded_file)


def inference_images(uploaded_file):
    image = Image.open(uploaded_file)
    # predict the image
    predict = model.predict(image)
    k = 0

    # plot boxes
    boxes = predict[0].boxes
    classes = boxes.cls.cpu().numpy()
    class_names = {
        0: "Buffalo",
        1: "Elephant",
        2: "Rhino",
        3: "Zebra",
        4: "Cheetah",
        5: "Fox",
        6: "Jaguar",
        7: "Tiger",
        8: "Lion",
        9: "Panda",
    }
    print(classes)

    print(classes)
    for i in classes:

        # print(class_names[classes[0]])
        print(classes)
        print("indidkfsflsjdflkjf")
        classes = classes[k:]
        k += 1

        subject = "Animal is detecting  alert the people"
        body = class_names[classes[0]]
        send_mail(subject, body)
    print("******************************************************")
    print(boxes)
    plotted = predict[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**No Detection**")

    # open the image.
    st.image(plotted, caption="Detected Image", width=600)
    logging.info("Detected Image")


def inference_video(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv.VideoCapture(temp_file.name)
    frame_count = 0
    if not cap.isOpened():
        st.error("Error opening video file.")

    frame_placeholder = st.empty()
    stop_placeholder = st.button("Stop")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % 2 == 0:
            # predict the frame
            predict = model.predict(frame, conf=0.75)
            boxes = predict[0].boxes
            classes = boxes.cls.cpu().numpy()
            print(classes)

            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
            # print(predict.keys())
            # plot boxes
            plotted = predict[0].plot()

            # Display the video
            frame_placeholder.image(plotted, channels="BGR", caption="Video Frame")

        # Clean up the temporary file
        if stop_placeholder:
            os.unlink(temp_file.name)
            break
    class_names = {
        0: "Buffalo",
        1: "Elephant",
        2: "Rhino",
        3: "Zebra",
        4: "Cheetah",
        5: "Fox",
        6: "Jaguar",
        7: "Tiger",
        8: "Lion",
        9: "Panda",
    }
    print(classes)
    classes = set(classes)
    classes = list(classes)
    for i in classes:

        print(class_names[classes[0]])
        subject = "Animal is detecting  alert the people"
        body = class_names[classes[0]]
        send_mail(subject, body)

    cap.release()


if __name__ == "__main__":
    main()
