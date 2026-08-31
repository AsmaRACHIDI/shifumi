"""Chargement et découpage du jeu de données rock_paper_scissors."""

import numpy as np

CLASS_NAMES = ("rock", "paper", "scissors")
CLASS_NAMES_FR = ("pierre", "feuille", "ciseaux")
NUM_CLASSES = 3
SEED = 42


def decoupe_stratifiee(X, y, ratio_validation=0.2, seed=SEED):
    """Découpe un jeu en entraînement et validation en préservant les proportions.

    Chaque classe est découpée séparément, ce qui garantit la même répartition
    dans les deux sous-ensembles quel que soit l'ordre initial des données.
    """
    rng = np.random.default_rng(seed)
    indices_train = []
    indices_val = []

    for classe in np.unique(y):
        indices = np.flatnonzero(y == classe)
        rng.shuffle(indices)
        coupe = int(len(indices) * ratio_validation)
        indices_val.append(indices[:coupe])
        indices_train.append(indices[coupe:])

    tr = np.concatenate(indices_train)
    val = np.concatenate(indices_val)
    rng.shuffle(tr)
    rng.shuffle(val)

    return X[tr], y[tr], X[val], y[val]
