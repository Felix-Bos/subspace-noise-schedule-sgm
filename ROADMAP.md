# ROADMAP — subspace-noise-schedule-sgm

> Feuille de route du projet, lisible par les humains **et** par les assistants IA (Claude, Copilot, Cursor…).
> Source de vérité unique : si un fichier, un message ou une IA contredit ce document, c'est ce document qui prime, sauf mise à jour explicite ci-dessous.
> Dernière mise à jour : 15/10/2026.

---

## 0. Comment utiliser ce fichier

- **Humains :** cochez les cases `[x]` au fur et à mesure, mettez à jour la section 9 (journal) à chaque point hebdomadaire.
- **IA :** lisez d'abord les sections 1, 3 et 4 avant d'écrire la moindre ligne de code. Une tâche n'est terminée que si ses **critères d'acceptation** sont remplis.
- Chaque tâche a un identifiant (`T-xx`), un responsable (**A** = théorie, **B** = expériences), une semaine cible et des critères d'acceptation.
- Pour donner une tâche à une IA : « Lis `ROADMAP.md` puis fais `T-xx` ».

---

## 1. Le projet en bref

**Cours :** Sylvain Le Corff, *Sampling for generative models* (M2 MS2A, Sorbonne).
**Livrable :** rapport LaTeX en français, 30–40 pages, type article (théorie + expériences), **rendu avant le 24/01/2027**.
**Binôme :** A = théorie, B = expériences.

**Article de départ :** Strasman, Ocello, Boyer, Le Corff, Lemaire, *An analysis of the noise schedule for score-based generative models*, TMLR 2025 (arXiv 2402.04650). Il prouve :

$$\mathrm{KL}(\pi_{\text{data}} \| \hat\pi) \le E_1 + E_2 + E_3$$

- $E_1$ : initialisation (bruit pur au lieu de la vraie loi à $T$)
- $E_2$ : erreur d'approximation du score
- $E_3$ : discrétisation (nombre fini de pas)

Le schedule $\beta(t)$ y est **scalaire** (même rythme de bruit dans toutes les directions).

**Notre extension :** un schedule **par sous-espace** ($J = 2$) :

$$\beta(t) = \beta_1(t)\,P_1 + \beta_2(t)\,P_2$$

- $P_1$ : projection sur les $k$ premières composantes ACP (forme globale) ;
- $P_2 = I - P_1$ (détails).

**Objectifs, par ordre de priorité :**

1. **Sûr :** mesurer exactement, dans le cas gaussien et à N fini, l'écart entre la borne de l'article et la vraie KL (EM et EI).
2. **Principal :** prouver une borne KL non asymptotique pour le schedule par sous-espace (version sûre : min / max des $\beta_j$).
3. **Bonus :** version fine — une condition sous laquelle le schedule différencié bat prouvablement le schedule unique.
4. **Images :** grilles couplées et trajectoires montrant l'effet (MNIST pour itérer, CIFAR-10 sur GPU pour le rapport).

**À ne pas refaire (déjà publié) :**

- Sun, Poor, Zhang (arXiv 2601.13602) : cas gaussien, N → ∞, sampler ODE, schedule optimal par mode propre (« loi tangente »), écart scalaire / anisotrope en forme close.
- Liu, Li, Cheng (arXiv 2602.19512) : apprentissage empirique de schedules matriciels, sans borne.

---

## 2. Arborescence du dépôt

```
subspace-noise-schedule-sgm/
├── ROADMAP.md              # ce fichier
├── README.md
├── LICENSE
├── report/                 # rapport LaTeX (français)
├── references/             # PDF des articles — IGNORÉ par git
├── src/
│   ├── noise_diff/         # code officiel de Strasman et al. — NE PAS MODIFIER
│   │   └── LICENSE         # leur licence MIT
│   └── subspace_sgm/       # notre package
│       ├── __init__.py
│       ├── conventions.py  # constantes et conventions (voir §3)
│       ├── schedules.py    # schedules scalaires et par sous-espace
│       ├── sde.py          # forward / backward
│       ├── samplers.py     # Euler–Maruyama (EM), intégrateur exponentiel (EI)
│       ├── gaussian_lab.py # KL / W2 exactes, récurrences linéaires gaussiennes
│       ├── bounds.py       # E1, E2, E3 (article + extension)
│       ├── subspaces.py    # projecteurs ACP P1, P2
│       ├── score_net.py    # réseau de score conditionné par (m1, m2)
│       ├── train.py
│       └── data/           # chargement MNIST, CIFAR-10
├── experiments/            # un script par figure : python experiments/<nom>.py
├── tests/                  # pytest
├── notebooks/              # exploration uniquement, jamais source d'une figure du rapport
├── data/                   # datasets — IGNORÉ par git
└── figures/                # sorties des scripts d'experiments/
```

---

## 3. Règles pour TOUTE contribution (humaine ou IA)

### 3.1 Règles absolues

1. **Ne jamais modifier `src/noise_diff/`.** On l'importe, on ne le corrige pas. Pour un comportement différent, on écrit notre version dans `src/subspace_sgm/`.
2. **Une seule source pour les conventions** : `src/subspace_sgm/conventions.py`. Aucun script ne redéfinit le facteur de $\beta$, $\sigma^2$, $m_t$ ou le sens du temps.
3. **Valider avant d'utiliser** : tout échantillonneur doit passer le test gaussien (T-05) avant d'être branché sur des images.
4. **Graines fixées** : chaque script d'`experiments/` accepte `--seed` et produit exactement la même figure à graine égale.
5. **Une figure = un script** dans `experiments/`, lançable en une commande, qui écrit dans `figures/`.
6. **Précision numérique** : le laboratoire gaussien tourne en **CPU, float64**. MPS (Mac) ne fait que du float32.
7. **Ne pas inventer** : si une formule de l'article est incertaine, laisser un `# TODO(vérifier §x de l'article)` plutôt que deviner.

### 3.2 Conventions à fixer (T-02)

À recopier **depuis l'article (§2) et le code `src/noise_diff/`**, puis à figer dans `conventions.py` :

| Élément | Valeur | Source |
|---|---|---|
| EDS forward | $dX_t = -\tfrac12\beta(t)X_t\,dt + \sqrt{\beta(t)}\,dW_t$ (à confirmer, y compris le rôle de $\sigma^2$) | Article §2 |
| $m_t$ | $\exp(-\tfrac12\int_0^t\beta)$ (à confirmer) | Article §2 |
| $\sigma_t^2$ | à recopier | Article §2 |
| Loi stationnaire | $\mathcal N(0, \sigma^2 I)$ (à confirmer) | Annexe B |
| Sens du temps backward | à recopier | Code `noise_diff/sampler.py` |
| Famille paramétrique de schedules (paramètre $a$) | à recopier | Article §5 / code |
| Horizon $T$, nombre de pas par défaut | à recopier | Article §5 |

### 3.3 Style de code

- Python ≥ 3.10, typage, docstrings en anglais ou en français (cohérent dans un même fichier).
- `ruff` actif sauf sur `src/noise_diff/` (`extend-exclude`).
- Pas de logique dans les notebooks qui serve au rapport.
- Tests dans `tests/`, lancés par `pytest -q`.

### 3.4 Instructions spécifiques aux assistants IA

- Lire §1, §3 et la tâche demandée avant de coder.
- Ne toucher qu'aux fichiers concernés par la tâche.
- Toujours ajouter ou mettre à jour un test quand on écrit une fonction numérique.
- Expliquer en fin de réponse : fichiers modifiés, tests ajoutés, critères d'acceptation remplis ou non.
- En cas de doute sur une convention mathématique : **s'arrêter et demander**, ne pas choisir arbitrairement.
- Ne jamais présenter un résultat numérique comme validé si le test correspondant n'a pas été exécuté.

---

## 4. Matériel

| Machine | Usage |
|---|---|
| Mac M3 (16 Go) | Développement, laboratoire gaussien (CPU float64), MNIST (`device="mps"`, `PYTORCH_ENABLE_MPS_FALLBACK=1`, `caffeinate -i` pour la nuit) |
| GPU (RTX 4000 Ada, 20 Go, ou plus) | Figures finales CIFAR-10 / CelebA 64×64, U-Net |
| PC professionnel | **Ne pas utiliser** |

---

## 5. Tâches

Statuts : `[ ]` à faire · `[~]` en cours · `[x]` fait · `[-]` abandonné

### Phase 0 — Mise en place (W1 : 12–18/10)

- [ ] **T-01 (A+B) — Réorganiser le dépôt**
  - Appliquer l'arborescence du §2 ; copier la licence MIT de l'article dans `src/noise_diff/LICENSE`.
  - `.gitignore` : `references/`, `data/`, `*.pt`, `*.ckpt`, `figures/*.png` (si on ne versionne pas les figures).
  - Exclure `src/noise_diff/` de ruff et de pre-commit.
  - ✅ Acceptation : `pytest -q` passe (même vide) ; `ruff check src/subspace_sgm` propre ; le dépôt est privé.

- [ ] **T-02 (A+B) — Lire l'article et figer les conventions**
  - Lire §2–4 et l'annexe B ensemble.
  - Remplir le tableau §3.2 et écrire `conventions.py`.
  - ✅ Acceptation : chaque ligne du tableau §3.2 a une valeur et une source (numéro d'équation ou fichier du code original).

- [ ] **T-03 (B) — Faire tourner le code officiel**
  - Lancer `notebooks/` ou l'exemple du code original sur un cas gaussien simple.
  - ✅ Acceptation : un notebook `notebooks/00_original_code_sanity.ipynb` reproduit au moins une courbe de l'article (KL ou W2 en fonction de $a$), à l'œil.

- [ ] **T-04 (A+B) — Administratif**
  - Vérifier les règles du cours sur l'IA générative et sur le partage de code.
  - Repérer la semaine où le chapitre 3 du poly est enseigné.
  - Vérifier la correspondance Thm 3.2 du poly ↔ Thm 3.1 de l'article, et le facteur $\max\{h\beta(T)/(4\sigma^2),1\}$ dans $E_3$.
  - ✅ Acceptation : réponses notées dans le journal (§9).

### Phase 1 — Base théorique et laboratoire gaussien (W2–W3 : 19/10–01/11)

- [ ] **T-05 (B) — Laboratoire gaussien, schedule scalaire** · `gaussian_lab.py`, `samplers.py`
  - Données $\mathcal N(0,\Sigma_0)$, score exact (linéaire).
  - Récurrence exacte de la covariance générée pour **EM** et **EI**, à N fini.
  - KL et W2 exactes entre $\mathcal N(0,\Sigma_0)$ et la loi générée.
  - ✅ Acceptation :
    - test : KL exacte ≈ KL estimée par Monte Carlo (≥ 10⁵ échantillons) à moins de 3 erreurs standard, pour 3 schedules et 3 valeurs de N ;
    - test : KL → (terme d'initialisation seul) quand N → ∞ ;
    - tout en float64.

- [ ] **T-06 (A) — Redémontrer la borne KL (scalaire)**
  - Ligne par ligne : Girsanov + traitement des données ; $E_1$ (Fokker–Planck, Sobolev logarithmique gaussienne, Grönwall) ; $E_3$ (Itô).
  - Rédiger en LaTeX dans `report/sections/borne_scalaire.tex`.
  - ✅ Acceptation : preuve complète, chaque étape justifiée, relue par B ; écarts avec le poly signalés.

- [ ] **T-07 (B) — Implémenter les termes de la borne** · `bounds.py`
  - $E_1$, $E_3$ (et $E_2$ pour mémoire ; $E_2 = 0$ avec score exact) dans le cas gaussien.
  - ✅ Acceptation : sur la famille de schedules de l'article, KL exacte ≤ $E_1 + E_3$ pour tous les cas testés ; figure `figures/gauss_slack_scalar.png` montrant le relâchement terme par terme.

- [ ] **T-08 (A) — Lire les concurrents**
  - Sun, Poor, Zhang (2601.13602) en priorité, puis Liu, Li, Cheng (2602.19512).
  - Recherche plus large : articles citant Strasman et al. traitant de schedules matriciels / anisotropes.
  - ✅ Acceptation : note d'une page dans `report/notes/etat_de_lart.md` : ce qu'ils prouvent, ce qu'ils ne prouvent pas, ce qui reste pour nous.

- [ ] **T-09 (A) — Énoncer l'extension**
  - Énoncé précis de la borne pour $\beta(t)=\beta_1 P_1+\beta_2 P_2$, version sûre (min / max).
  - Envoyer l'énoncé à B **avant** de prouver.
  - ✅ Acceptation : énoncé écrit dans `report/sections/extension.tex` avec hypothèses explicites.

### Phase 2 — Extension et réseau (W4–W6 : 02–22/11)

- [ ] **T-10 (B) — Laboratoire gaussien, schedule par sous-espace** · `schedules.py`, `subspaces.py`, `gaussian_lab.py`
  - Généraliser T-05 à $\beta_1 P_1 + \beta_2 P_2$.
  - ✅ Acceptation :
    - test : $\beta_1=\beta_2$ redonne exactement les résultats scalaires de T-05 ;
    - test Monte Carlo comme T-05.

- [ ] **T-11 (B) — Tester numériquement l'énoncé de A**
  - Balayer : spectres de $\Sigma_0$, choix de $k$, couples ($\beta_1$, $\beta_2$), N.
  - ✅ Acceptation : rapport court « tient / contre-exemple » dans le journal, avec la figure `figures/gauss_subspace_bound_check.png`.

- [ ] **T-12 (A) — Prouver la version sûre**
  - ✅ Acceptation : preuve complète dans `extension.tex`, vérifiée numériquement par T-11.

- [ ] **T-13 (B) — Réseau de score conditionné par (m₁, m₂)** · `score_net.py`, `train.py`
  - Entrées : image bruitée, $(m_{t,1}, m_{t,2})$. Un seul entraînement pour toute la famille de schedules.
  - MNIST sur Mac d'abord.
  - ✅ Acceptation : échantillons MNIST reconnaissables avec le schedule scalaire de l'article ; checkpoint sauvegardé hors git ; commande d'entraînement documentée dans le README.

- [ ] **T-14 (B) — Premières images couplées** · `experiments/mnist_grid.py`
  - Même point initial et mêmes incréments gaussiens pour tous les schedules.
  - Colonnes : scalaire de l'article, différencié (forme rapide / détails lents, et l'inverse), loi tangente de Sun et al.
  - ✅ Acceptation : `figures/mnist_grid.png` reproductible avec `--seed`.

- [ ] **T-15 (A) — Tenter la version fine**
  - Hypothèse de faible dépendance entre sous-espaces, découpage terme par terme.
  - ✅ Acceptation : soit une preuve, soit une note expliquant précisément où ça bloque.

- [ ] **T-16 (B) — Trajectoires et nombre de pas** · `experiments/mnist_trajectories.py`, `experiments/mnist_nsteps.py`
  - Instantanés $x_{t_k}$ ; comparaison à 10, 20, 50 pas.
  - ✅ Acceptation : deux figures reproductibles.

### 🚦 Point de décision — 22/11

- [ ] **T-17 (A+B) — Décision**
  - On continue la version fine **si** : KL exacte sous la borne partout (T-07, T-11), version sûre prouvée (T-12), grilles MNIST exploitables (T-14).
  - **Sinon** : on garde version sûre + mesure gaussienne, on abandonne la version fine, on avance la rédaction.
  - **Repli ultime** (si T-12 échoue) : early stopping sous hypothèse d'information de Fisher finie — vérifier l'originalité avant.
  - ✅ Acceptation : décision écrite dans le journal, tâches suivantes ajustées.

### Phase 3 — Finalisation (W7–W8 : 23/11–06/12)

- [ ] **T-18 (A) — Finir la preuve ou basculer sur le repli**
- [ ] **T-19 (A, optionnel) — Borne W₂** (couplage synchrone), seulement s'il reste du temps ; sinon énoncée et citée.
- [ ] **T-20 (B) — Figures finales sur GPU** · `experiments/cifar_*.py`
  - CIFAR-10 (ou CelebA 64×64), U-Net, J = 2.
  - ✅ Acceptation : mêmes figures que T-14 et T-16 en CIFAR-10, reproductibles.
- [ ] **T-21 (B) — Robustesse**
  - Relancer chaque expérience 3 à 5 graines ; moyenne ± écart-type où c'est pertinent.
  - ✅ Acceptation : tableau récapitulatif dans `report/sections/experiences.tex`.

### Phase 4 — Rédaction (W9–W16 : 07/12–24/01)

- [ ] **T-22 (A) — Rédiger** : cadre, borne scalaire, extension, annexe des preuves.
- [ ] **T-23 (B) — Rédiger** : laboratoire gaussien, expériences, limites, annexe de reproductibilité, déclaration d'usage des LLM.
- [ ] **T-24 (A+B) — Relecture croisée** (Noël).
- [ ] **T-25 (A+B) — Brouillon complet** : 10/01.
- [ ] **T-26 (A+B) — Réduction à 30–40 pages, vérification de toutes les références, rendu avant le 24/01.**

---

## 6. Calendrier résumé

| Semaine | Dates | A | B |
|---|---|---|---|
| W1 | 12–18/10 | T-01, T-02, T-04 | T-01, T-02, T-03 |
| W2 | 19–25/10 | T-06 | T-05 |
| W3 | 26/10–01/11 | T-08, T-09 | T-07 |
| W4 | 02–08/11 | T-12 | T-10, T-13 |
| W5 | 09–15/11 | T-12 | T-11, T-14 |
| W6 | 16–22/11 | T-15, **T-17** | T-16, **T-17** |
| W7–W8 | 23/11–06/12 | T-18, T-19 | T-20, T-21 |
| W9–W10 | 07–20/12 | T-22 | T-23 |
| W11–W12 | 21/12–03/01 | T-24 (50 %) | T-24 (50 %) |
| W13 | 04–10/01 | T-25 | T-25 |
| W14–W16 | 11–31/01 | T-26 | T-26 |

Charge visée : ≈ 10 h par personne et par semaine (≈ 196 h au total, A ≈ 108 h, B ≈ 88 h).

---

## 7. Structure du rapport

Titre : *Noise schedules par sous-espace pour les modèles génératifs à score : bornes non asymptotiques et expériences*

1. Introduction
2. Cadre (notations du poly)
3. La borne de Strasman et al. (redémontrée)
4. Laboratoire gaussien
5. Schedules par sous-espace
6. Expériences
7. Limites
8. Perspectives
9. Annexes : preuves, implémentation, reproductibilité, usage des LLM

---

## 8. Références

- Strasman et al., TMLR 2025 — https://arxiv.org/abs/2402.04650
- Code officiel (MIT) — https://github.com/StanislasStrasman/Noise_Schedule_for_Score-based_Generative_Models
- Strasman et al., *On Forgetting and Stability of Score-based Generative Models* — https://arxiv.org/abs/2601.21868
- Sun, Poor, Zhang, *A Gaussian Perspective for Distributional Discrepancy in Generative Diffusion Models* — https://arxiv.org/abs/2601.13602
- Liu, Li, Cheng, *Variational Trajectory Optimization of Anisotropic Diffusion Schedules* — https://arxiv.org/abs/2602.19512
- Poly : S. Le Corff, *Sampling for generative models — Theory and practical applications*

---

## 9. Journal

> Une entrée par point hebdomadaire (30 min). Format : date · qui · fait · bloqué · décisions.

| Date | Qui | Fait | Bloqué | Décisions |
|---|---|---|---|---|
| 15/10/2026 | A+B | Choix du sujet, structure du dépôt, roadmap | — | Dépôt `subspace-noise-schedule-sgm`, J = 2, point de décision unique le 22/11 |
| | | | | |

**Rôles :** A = _______ · B = _______
