import numpy as np

from shifumi.data import decoupe_stratifiee


def test_le_decoupage_preserve_les_proportions_et_ne_duplique_rien():
    # 90 images factices minuscules, 30 par classe, dans l'ordre.
    X = np.arange(90).reshape(90, 1, 1, 1)
    y = np.repeat([0, 1, 2], 30)

    X_tr, y_tr, X_val, y_val = decoupe_stratifiee(X, y, ratio_validation=0.2)

    # Stratification : 6 exemples de chaque classe en validation.
    assert list(np.bincount(y_val)) == [6, 6, 6]

    # Aucune perte : tous les exemples sont quelque part.
    assert len(X_tr) + len(X_val) == 90
