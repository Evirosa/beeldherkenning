import cv2 as cv
import numpy as np

frame = cv.imread("koekje2.jpg")
print(frame.shape)

if frame is None:
    print("Foto kon niet worden geladen")
    exit()

    # Convert BGR to HSV
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_yellow = np.array([12, 50, 50])
upper_yellow = np.array([45, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
mask_red = cv.inRange(hsv, lower_red, upper_red)
mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
mask_green = cv.inRange(hsv, lower_green, upper_green)
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
mask_all = cv.bitwise_or(mask_red, cv.bitwise_or(mask_yellow, cv.bitwise_or(mask_green, mask_blue)))
res = cv.bitwise_and(frame, frame, mask=mask_all)
drempel = 200

kleuren = {
"rood": mask_red,
"geel": mask_yellow,
"groen": mask_green,
"blauw": mask_blue,
}
 
print("Resultaat kleurdetectie:")
gevonden_kleuren = []
for naam, mask in kleuren.items():
    aantal_pixels = cv.countNonZero(mask)
    aanwezig = aantal_pixels > drempel
    print(f"  {naam}: {'JA' if aanwezig else 'nee'} ({aantal_pixels} pixels)")
    if aanwezig:
        gevonden_kleuren.append(naam)
 
if gevonden_kleuren:
    print("\nDe foto bevat: " + ", ".join(gevonden_kleuren))
else:
    print("\nGeen van de gezochte kleuren gevonden.")
 
# --- Vensters tonen tot ESC wordt ingedrukt ---
while True:
    cv.imshow('frame', frame)
    cv.imshow('mask', mask_all)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:  # ESC
        break
 
cv.destroyAllWindows()