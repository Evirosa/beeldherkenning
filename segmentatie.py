"""
segmentatie.py
Module voor het scheiden van het koekje (voorgrond) van een effen,
lichte achtergrond.

Waarom nodig: de kleurherkenning in hsv.py telt gewoon pixels. Als de
hele foto (inclusief achtergrond) wordt meegenomen, wint de achtergrond
vaak van het koekje omdat hij simpelweg meer pixels beslaat. Door eerst
een algemeen voorgrondmasker te maken en de kleurherkenning daarna
alleen bínnen dat gebied uit te voeren, telt de achtergrond niet meer
mee.

Aanname: de achtergrond is wit/grijs (lage kleurverzadiging, hoge
helderheid). Het koekje zelf heeft altijd meer verzadiging (bruin, geel,
rood) of is duidelijk minder helder dan de achtergrond.
"""

import cv2
import numpy as np


def maak_voorgrondmasker(bgr_afbeelding, verzadigings_drempel=40, helderheids_drempel=200):
    """
    Maakt een ruw binair masker van het voorwerp (koekje) ten opzichte
    van een effen, lichte achtergrond.

    :param bgr_afbeelding: originele afbeelding in BGR
    :param verzadigings_drempel: S-waarde (HSV) waarboven een pixel als
        "voorgrond" geldt (kleurrijke pixels horen bij het koekje)
    :param helderheids_drempel: V-waarde (HSV) waaronder een pixel ook
        als "voorgrond" geldt (donkere randen/schaduw van het koekje,
        ook als de kleur zelf weinig verzadiging heeft)
    :return: binair masker (uint8, 0/255), 255 = koekje/voorgrond
    """
    hsv = cv2.cvtColor(bgr_afbeelding, cv2.COLOR_BGR2HSV)
    verzadiging = hsv[:, :, 1]
    helderheid = hsv[:, :, 2]

    is_verzadigd = verzadiging > verzadigings_drempel
    is_donker = helderheid < helderheids_drempel

    voorgrond = np.logical_or(is_verzadigd, is_donker)
    voorgrondmasker = np.where(voorgrond, 255, 0).astype(np.uint8)

    return voorgrondmasker