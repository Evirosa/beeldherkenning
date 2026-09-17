import os
import cv2
import numpy as np

# Definieer de video's en de bijbehorende labels
VIDEOS = [
    {"pad": "stroopwafels.mp4", "label": "stroopwafel"},
    {"pad": "kokosmakroon.mp4", "label": "kokosmakroon"},
    {"pad": "kano.mp4", "label": "kano"}
]

DOEL_AANTAL = 1000
OUTPUT_HOOFDMAP = "dataset"

for item in VIDEOS:
    video_pad = item["pad"]
    label = item["label"]
    
    cap = cv2.VideoCapture(video_pad)
    if not cap.isOpened():
        print(f"Kon video niet openen: {video_pad}")
        continue

    totaal_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Bezig met '{label}': {totaal_frames} frames beschikbaar...")

    if totaal_frames < DOEL_AANTAL:
        print(f"Let op: {video_pad} heeft slechts {totaal_frames} frames. Pas DOEL_AANTAL aan.")
        cap.release()
        continue

    # Maak een doelmap per categorie (bijv. dataset/stroopwafel/)
    uitvoermap = os.path.join(OUTPUT_HOOFDMAP, label)
    os.makedirs(uitvoermap, exist_ok=True)

    # Kies precies 1.000 unieke, gelijkmatig verdeelde frame-indices
    geselecteerde_indices = set(np.linspace(0, totaal_frames - 1, DOEL_AANTAL, dtype=int))

    huidige_frame_idx = 0
    opgeslagen_teller = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if huidige_frame_idx in geselecteerde_indices:
            opgeslagen_teller += 1
            # Bestandsnaam met voorloopnullen (bijv. stroopwafel_0001.jpg)
            bestandsnaam = f"{label}_{opgeslagen_teller:04d}.jpg"
            volledig_pad = os.path.join(uitvoermap, bestandsnaam)
            
            # Opslaan met hoge JPG-kwaliteit
            cv2.imwrite(volledig_pad, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        huidige_frame_idx += 1

    cap.release()
    print(f"Klaar: {opgeslagen_teller} afbeeldingen opgeslagen in '{uitvoermap}'.")