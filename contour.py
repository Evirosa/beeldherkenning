"""
contouren.py
Functiemodule voor het vinden van contouren in een binair masker
en het bepalen van vorm en grootte van het grootste object daarin.
"""

import cv2


def vind_grootste_contour(masker):
    """
    Zoekt alle externe contouren in het masker en geeft de grootste terug.
    Er wordt uitgegaan van één object (koekje) per foto.

    :param masker: binair masker (uint8)
    :return: contour (numpy array) van het grootste object, of None
    """
    contouren, _ = cv2.findContours(
        masker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contouren:
        return None

    grootste_contour = max(contouren, key=cv2.contourArea)
    return grootste_contour


def bepaal_vorm(contour):
    """
    Bepaalt of een contour rond, vierkant of rechthoekig is, aan de hand
    van het aantal hoekpunten na polygoonbenadering en de verhouding
    tussen breedte en hoogte van de omvattende rechthoek.

    :param contour: contour zoals gevonden door cv2.findContours
    :return: vormnaam ("rond", "vierkant", "rechthoek" of "onbekend")
    """
    omtrek = cv2.arcLength(contour, True)
    # 3% van de omtrek als benaderingsnauwkeurigheid is een gangbare
    # vuistregel om ruis in de contourrand te onderdrukken.
    benadering = cv2.approxPolyDP(contour, 0.03 * omtrek, True)
    aantal_hoeken = len(benadering)

    _, _, breedte, hoogte = cv2.boundingRect(contour)
    verhouding = breedte / float(hoogte)

    if aantal_hoeken >= 8:
        vorm = "rond"
    elif aantal_hoeken == 4 and 0.9 <= verhouding <= 1.1:
        vorm = "vierkant"
    elif aantal_hoeken == 4:
        vorm = "rechthoek"
    else:
        vorm = "onbekend"

    return vorm


def bepaal_grootte(contour):
    """
    Bepaalt de oppervlakte van de contour in pixels.

    :param contour: contour zoals gevonden door cv2.findContours
    :return: oppervlakte in pixels (float)
    """
    return cv2.contourArea(contour)