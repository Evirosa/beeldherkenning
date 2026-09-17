"""
hsv.py
Functiemodule voor kleurherkenning met behulp van de HSV-kleurruimte.

HSV wordt gebruikt in plaats van RGB/BGR omdat de Hue-waarde (kleurtint)
vrijwel onafhankelijk is van lichtintensiteit. Dit maakt kleurherkenning
robuuster tegen schaduw en belichtingsverschillen.
"""

import cv2
import numpy as np

# Kleurgrenzen in HSV-formaat (Hue: 0-179, Saturation: 0-255, Value: 0-255)
# Deze grenzen zijn experimenteel bepaald en moeten evt. gekalibreerd
# worden op de eigen camera/belichting.
KLEURGRENZEN = {
    "bruin": [(np.array([5, 50, 20]), np.array([20, 255, 200]))],
    "geel": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
    "wit": [(np.array([0, 0, 180]), np.array([180, 40, 255]))],
    "rood": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255])),
    ],
}


def bepaal_kleurmasker(hsv_afbeelding, kleurnaam):
    """
    Maakt een binair masker voor een gegeven kleurnaam.

    :param hsv_afbeelding: afbeelding in HSV-kleurruimte (numpy array)
    :param kleurnaam: sleutel uit KLEURGRENZEN
    :return: binair masker (numpy array, waarden 0 of 255)
    """
    if kleurnaam not in KLEURGRENZEN:
        raise ValueError(f"Onbekende kleurnaam: {kleurnaam}")

    masker_totaal = np.zeros(hsv_afbeelding.shape[:2], dtype=np.uint8)

    # Sommige kleuren (zoals rood) liggen op twee plekken op de Hue-cirkel,
    # daarom kunnen er meerdere grensparen per kleur zijn.
    for ondergrens, bovengrens in KLEURGRENZEN[kleurnaam]:
        deelmasker = cv2.inRange(hsv_afbeelding, ondergrens, bovengrens)
        masker_totaal = cv2.bitwise_or(masker_totaal, deelmasker)

    return masker_totaal


def herken_dominante_kleur(bgr_afbeelding, zoekmasker=None):
    """
    Bepaalt welke gedefinieerde kleur het meest voorkomt in de afbeelding.

    :param bgr_afbeelding: originele afbeelding in BGR (output van cv2.imread)
    :param zoekmasker: optioneel binair masker (uint8, 0/255) dat aangeeft
        binnen welk gebied gezocht mag worden (bv. alleen het koekje,
        zonder achtergrond). Als dit meegegeven wordt, worden pixels
        buiten dit gebied volledig genegeerd bij het tellen.
    :return: tuple (kleurnaam: str of None, masker: np.ndarray of None)
    """
    hsv = cv2.cvtColor(bgr_afbeelding, cv2.COLOR_BGR2HSV)

    beste_kleur = None
    beste_masker = None
    grootste_aantal_pixels = 0

    for kleurnaam in KLEURGRENZEN:
        masker = bepaal_kleurmasker(hsv, kleurnaam)

        if zoekmasker is not None:
            masker = cv2.bitwise_and(masker, zoekmasker)

        aantal_pixels = cv2.countNonZero(masker)

        if aantal_pixels > grootste_aantal_pixels:
            grootste_aantal_pixels = aantal_pixels
            beste_kleur = kleurnaam
            beste_masker = masker

    return beste_kleur, beste_masker