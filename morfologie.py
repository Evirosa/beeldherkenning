

import cv2

def filter_ruis(masker, kernelgrootte=5, iteraties=2):
    """
    Verwijdert ruis uit een binair masker met opening en closing.

    - Opening (erosie gevolgd door dilatatie) verwijdert kleine losse
      witte pixels (ruis) buiten het object.
    - Closing (dilatatie gevolgd door erosie) vult kleine gaatjes
      binnen het object op.

    :param masker: binair masker (uint8, waarden 0/255)
    :param kernelgrootte: afmeting van het structurerend element (oneven getal)
    :param iteraties: aantal keer dat elke bewerking wordt toegepast
    :return: gefilterd binair masker (np.ndarray)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernelgrootte, kernelgrootte)
    )

    masker_geopend = cv2.morphologyEx(
        masker, cv2.MORPH_OPEN, kernel, iterations=iteraties
    )
    masker_gesloten = cv2.morphologyEx(
        masker_geopend, cv2.MORPH_CLOSE, kernel, iterations=iteraties
    )

    return masker_gesloten