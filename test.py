import cv2
import numpy as np
import time

# Open de webcam (0 = standaardcamera)
cap = cv2.VideoCapture(0)

# Tijd waarop we voor het laatst hebben gecontroleerd
last_check = time.time()

while True:
    # Lees een frame van de camera
    ret, frame = cap.read()

    if not ret:
        print("Camera kon niet worden geopend")
        break

    # Zet het beeld om naar HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Eerste rode kleurgebied
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # Tweede rode kleurgebied
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Masks maken
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Masks combineren
    mask = mask1 + mask2

    # Huidige tijd
    current_time = time.time()

    # Elke 10 seconden controleren
    if current_time - last_check >= 10:

        # Tel hoeveel rode pixels er zijn
        red_pixels = cv2.countNonZero(mask)

        # Controleer of er genoeg rood aanwezig is
        if red_pixels > 1000:
            print("🔴 Rood object gedetecteerd!")
        else:
            print("⚪ Geen rood object gedetecteerd.")

        # Timer opnieuw starten
        last_check = current_time

    # Laat de camera zien
    cv2.imshow("Camera", frame)

    # Laat zien wat als rood wordt herkend
    cv2.imshow("Rood detectie", mask)

    # Stop met q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Camera afsluiten
cap.release()
cv2.destroyAllWindows()