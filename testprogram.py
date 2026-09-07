import cv2 as cv
import numpy as np


def beeldbewerking(frame):

    # ---------------------------------------------------------
    # Origineel
    # ---------------------------------------------------------

    origineel = frame.copy()

    # ---------------------------------------------------------
    # Grayscale
    # ---------------------------------------------------------

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # HSV
    # ---------------------------------------------------------

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # ---------------------------------------------------------
    # Threshold
    # ---------------------------------------------------------

    _, threshold = cv.threshold(
        gray,
        120,
        255,
        cv.THRESH_BINARY
    )

    # ---------------------------------------------------------
    # Canny
    # ---------------------------------------------------------

    canny = cv.Canny(
        gray,
        50,
        150
    )

    # ---------------------------------------------------------
    # Morphologie
    # ---------------------------------------------------------

    kernel = np.ones((5, 5), np.uint8)

    morphologie = cv.morphologyEx(
        threshold,
        cv.MORPH_OPEN,
        kernel
    )

    # ---------------------------------------------------------
    # Contouren
    # ---------------------------------------------------------

    contouren = frame.copy()

    contours, _ = cv.findContours(
        threshold,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        oppervlakte = cv.contourArea(contour)

        # Kleine ruis negeren
        if oppervlakte > 500:

            cv.drawContours(
                contouren,
                [contour],
                -1,
                (0, 255, 0),
                2
            )

            x, y, w, h = cv.boundingRect(contour)

            cv.rectangle(
                contouren,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

    # ---------------------------------------------------------
    # HSV kleurfilter
    # ---------------------------------------------------------

    # Voorbeeldwaarden
    # Deze kunnen we later aanpassen aan jouw koekjes.

    lower = np.array([5, 50, 30])
    upper = np.array([30, 255, 255])

    mask = cv.inRange(
        hsv,
        lower,
        upper
    )

    kleurfilter = cv.bitwise_and(
        frame,
        frame,
        mask=mask
    )

    # ---------------------------------------------------------
    # Hough cirkels
    # ---------------------------------------------------------

    cirkels = frame.copy()

    blur = cv.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    circles = cv.HoughCircles(
        blur,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=40,
        minRadius=10,
        maxRadius=200
    )

    if circles is not None:

        circles = np.round(
            circles[0, :]
        ).astype("int")

        for x, y, r in circles:

            cv.circle(
                cirkels,
                (x, y),
                r,
                (0, 255, 0),
                2
            )

            cv.circle(
                cirkels,
                (x, y),
                2,
                (0, 0, 255),
                3
            )

    # ---------------------------------------------------------
    # Alles geschikt maken voor display
    # ---------------------------------------------------------

    gray = cv.cvtColor(
        gray,
        cv.COLOR_GRAY2BGR
    )

    threshold = cv.cvtColor(
        threshold,
        cv.COLOR_GRAY2BGR
    )

    canny = cv.cvtColor(
        canny,
        cv.COLOR_GRAY2BGR
    )

    morphologie = cv.cvtColor(
        morphologie,
        cv.COLOR_GRAY2BGR
    )

    mask = cv.cvtColor(
        mask,
        cv.COLOR_GRAY2BGR
    )

    # ---------------------------------------------------------
    # Afbeeldingen kleiner maken
    # ---------------------------------------------------------

    schaal = 0.5

    origineel = cv.resize(
        origineel,
        None,
        fx=schaal,
        fy=schaal
    )

    gray = cv.resize(
        gray,
        None,
        fx=schaal,
        fy=schaal
    )

    threshold = cv.resize(
        threshold,
        None,
        fx=schaal,
        fy=schaal
    )

    canny = cv.resize(
        canny,
        None,
        fx=schaal,
        fy=schaal
    )

    morphologie = cv.resize(
        morphologie,
        None,
        fx=schaal,
        fy=schaal
    )

    contouren = cv.resize(
        contouren,
        None,
        fx=schaal,
        fy=schaal
    )

    kleurfilter = cv.resize(
        kleurfilter,
        None,
        fx=schaal,
        fy=schaal
    )

    cirkels = cv.resize(
        cirkels,
        None,
        fx=schaal,
        fy=schaal
    )

    # ---------------------------------------------------------
    # Titels toevoegen
    # ---------------------------------------------------------

    cv.putText(
        origineel,
        "ORIGINEEL",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv.putText(
        gray,
        "GRAYSCALE",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv.putText(
        threshold,
        "THRESHOLD",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv.putText(
        canny,
        "CANNY",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv.putText(
        morphologie,
        "MORPHOLOGIE",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv.putText(
        contouren,
        "CONTOUREN",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv.putText(
        kleurfilter,
        "HSV KLEURFILTER",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv.putText(
        cirkels,
        "HOUGH CIRKELS",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ---------------------------------------------------------
    # 4 x 2 afbeeldingen naast elkaar
    # ---------------------------------------------------------

    boven = np.hstack([
        origineel,
        gray,
        threshold,
        canny
    ])

    onder = np.hstack([
        morphologie,
        contouren,
        kleurfilter,
        cirkels
    ])

    resultaat = np.vstack([
        boven,
        onder
    ])

    return resultaat


def main():

    # ---------------------------------------------------------
    # Camera openen
    # ---------------------------------------------------------

    camera = cv.VideoCapture(0)

    if not camera.isOpened():

        print("Fout: camera kon niet worden geopend.")
        return

    print("Camera gestart.")
    print("Druk op Q om te stoppen.")

    while True:

        # Frame uit camera halen
        ret, frame = camera.read()

        if not ret:

            print("Fout bij het lezen van de camera.")
            break

        # Beeldbewerkingen uitvoeren
        resultaat = beeldbewerking(frame)

        # Resultaat tonen
        cv.imshow(
            "Beeldherkenning - Testprogramma",
            resultaat
        )

        # Q = stoppen
        if cv.waitKey(1) & 0xFF == ord("q"):

            break

    # Camera netjes afsluiten toch even anders 
    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()