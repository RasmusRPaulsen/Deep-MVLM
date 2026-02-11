# Methodology (Thesis-ready draft)

## Study design and data

A cross-sectional dataset of $n=60$ subjects was used. For each subject, 23 predefined soft-tissue facial landmarks were annotated by an orthodontist in a fixed order (Tr, G, N, Prn, Col, Sn, Phi, Ls, Stm, Li, Pg, Gn, Me, Ex\_R, End\_R, End\_L, Ex\_L, Al\_R, Al\_L, Ch\_R, Ch\_L, Go\_R, Go\_L). Each landmark is represented by 3D coordinates $(x, y, z)$. The target outcome was orthodontic skeletal class with three categories: Class I, Class II, and Class III.

Given the limited sample size, the analysis was designed to (i) minimize overfitting, (ii) avoid optimistic bias via nested validation, and (iii) maximize interpretability and clinical plausibility.

## Preprocessing and coordinate handling

### Raw coordinates vs geometric features

Raw Cartesian coordinates are generally not directly comparable across individuals unless the coordinate system is standardized (translation, rotation, and scale). Using raw coordinates without alignment can cause the model to learn scanner- or pose-specific information rather than anatomy.

Accordingly, the primary analysis used engineered geometric features that are invariant (or weakly sensitive) to rigid-body transformations: distances, angles, ratios, and signed projections in a subject-specific anatomical reference frame.

### Subject-specific local anatomical frame

To retain clinically relevant *directional* (antero–posterior) information while remaining robust to the global coordinate system, a local right-handed orthonormal frame was constructed per subject:

- Origin: $\mathbf{o} = \mathbf{N}$ (nasion)
- Left–right axis: $\hat{\mathbf{x}} = \frac{\mathbf{Ex_R} - \mathbf{Ex_L}}{\|\mathbf{Ex_R} - \mathbf{Ex_L}\|}$
- Superior–inferior seed axis: $\hat{\mathbf{y}}_0 = \frac{\mathbf{Tr} - \mathbf{Me}}{\|\mathbf{Tr} - \mathbf{Me}\|}$
- Antero–posterior axis: $\hat{\mathbf{z}} = \frac{\hat{\mathbf{x}} \times \hat{\mathbf{y}}_0}{\|\hat{\mathbf{x}} \times \hat{\mathbf{y}}_0\|}$

The $\hat{\mathbf{z}}$ axis was oriented to be anterior-positive by enforcing $\hat{\mathbf{z}} \cdot (\mathbf{Prn} - \mathbf{N}) > 0$. Then the vertical axis was re-orthogonalized as $\hat{\mathbf{y}} = \frac{\hat{\mathbf{z}} \times \hat{\mathbf{x}}}{\|\hat{\mathbf{z}} \times \hat{\mathbf{x}}\|}$.

Coordinates were normalized for size using biocular width $s = \|\mathbf{Ex_R}-\mathbf{Ex_L}\|$. The local normalized coordinates of a landmark point $\mathbf{p}$ were

$$
\mathbf{p}' = \frac{1}{s}
\begin{bmatrix}
(\mathbf{p}-\mathbf{o})\cdot\hat{\mathbf{x}}\\
(\mathbf{p}-\mathbf{o})\cdot\hat{\mathbf{y}}\\
(\mathbf{p}-\mathbf{o})\cdot\hat{\mathbf{z}}
\end{bmatrix}.
$$

The main model used invariant distances/angles/ratios and selected signed sagittal projections (i.e., $z'$ components and differences), which directly encode relative protrusion/retrusion.

## Feature engineering (clinically motivated)

A priori features were defined to reflect sagittal facial profile and proportional relationships:

### Distance-based features (size-normalized)

Examples (computed in the local frame after scaling by $s$):

- Facial heights: $d(N,Me)$, $d(N,Sn)$, $d(Sn,Me)$
- Nose dimensions: $d(N,Prn)$, $d(Sn,Prn)$, $d(Col,Sn)$
- Chin prominence: $d(Sn,Pg)$, $d(N,Pg)$
- Transverse widths: $d(Ch_R,Ch_L)$, $d(Al_R,Al_L)$, $d(End_R,End_L)$, $d(Go_R,Go_L)$

### Angle-based features

Angles were selected to represent convexity and soft-tissue profile:

- Soft-tissue convexity at subnasale: $\angle(G,Sn,Pg)$
- Facial convexity at nasion: $\angle(G,N,Pg)$
- Profile angle at subnasale: $\angle(N,Sn,Pg)$
- Nasolabial angle: $\angle(Col,Sn,Ls)$
- Mentolabial (approx.) angle: $\angle(Li,Phi,Pg)$

### Ratio features

Dimensionless ratios were used to capture proportions:

- Lower-face proportion: $\frac{d(Sn,Me)}{d(N,Me)}$
- Nose-to-face: $\frac{d(N,Prn)}{d(N,Me)}$
- Chin-to-face: $\frac{d(Sn,Pg)}{d(N,Me)}$
- Mouth-to-nose width: $\frac{d(Ch_R,Ch_L)}{d(Al_R,Al_L)}$

### Signed sagittal features

In the anatomical frame, anterior–posterior differences (anterior-positive) were included:

- $z'(Pg)-z'(Sn)$, $z'(Prn)-z'(Sn)$, $z'(Ls)-z'(Sn)$, $z'(Li)-z'(Sn)$

These are expected to reflect relative soft-tissue retrusion/protrusion patterns associated with Class II and Class III.

## Mathematical definitions

### 3D Euclidean distance

For two landmarks $\mathbf{a}, \mathbf{b} \in \mathbb{R}^3$:

$$
d(\mathbf{a},\mathbf{b}) = \|\mathbf{a}-\mathbf{b}\|_2 = \sqrt{(a_x-b_x)^2 + (a_y-b_y)^2 + (a_z-b_z)^2}.
$$

### 3D angle between three landmarks

For landmarks $\mathbf{a}, \mathbf{b}, \mathbf{c}$ defining angle $\angle ABC$ at $\mathbf{b}$:

$$
\theta = \arccos\left(\frac{(\mathbf{a}-\mathbf{b})\cdot(\mathbf{c}-\mathbf{b})}{\|\mathbf{a}-\mathbf{b}\|\,\|\mathbf{c}-\mathbf{b}\|}\right).
$$

### Ratio features

For any two positive-valued features $u$ and $v$:

$$
r = \frac{u}{v}.
$$

Ratios were preferred when they represent clinically meaningful proportions and reduce dependence on overall size.

## Model selection (small-sample rationale)

Given $n=60$, high-capacity models (e.g., deep learning) are inappropriate. A regularized multinomial logistic regression with elastic-net penalty was selected because it:

1. Controls model complexity via $\ell_1/\ell_2$ regularization,
2. Produces probabilistic outputs and calibrated decision boundaries,
3. Supports interpretable coefficients and feature ranking.

All continuous predictors were standardized within the training folds.

## Validation strategy

To prevent optimistic bias, **nested repeated stratified cross-validation** was used:

- Outer loop: repeated stratified $K$-fold (default $K=5$) repeated multiple times (default 20 repeats) to estimate generalization.
- Inner loop: stratified $K$-fold (default 3-fold) for hyperparameter tuning.

Hyperparameters tuned: inverse regularization strength $C$ and elastic-net mixing parameter $\alpha$ (implemented as `l1_ratio`).

## Overfitting safeguards and statistical checks

- Nested CV to separate model selection from performance estimation.
- Regularization and feature standardization.
- Permutation importance for model interpretation (computed on the final refit model; not used to claim performance).

Optionally, a label-permutation test can be applied to evaluate whether performance exceeds chance under the null hypothesis of no association.

## Performance metrics (multi-class)

Performance was reported using:

- Accuracy
- Balanced accuracy
- Macro-averaged F1 score (primary)
- Weighted F1 score
- Matthews correlation coefficient (MCC)
- One-vs-rest macro ROC-AUC (when probabilistic outputs available)

Confusion matrices were reported to characterize misclassification structure.

## Gender incorporation

Gender was included as a covariate (binary indicator) rather than used for stratified subgroup modeling, because stratified modeling would substantially reduce the effective sample size. The effect of gender can be assessed by comparing performance with and without the covariate, and via permutation importance.

## Model interpretation

Interpretation emphasized:

- Permutation importance (macro-F1 drop under permutation)
- Direction and magnitude of standardized logistic regression coefficients (with appropriate caution due to multicollinearity)

SHAP-style explanations can be considered, but should be used cautiously in very small samples and only if computationally stable.

## Reproducible implementation

The reference implementation is provided in the `class-identifier` folder:

- `geom_features.py`: feature definitions and formulas
- `run_pipeline.py`: nested CV training/evaluation and permutation importance

Run:

```bash
/Users/eck/Documents/yüksek-lisans/MorphoMark/.venv/bin/python class-identifier/run_pipeline.py
```

Outputs are written under `class-identifier/outputs/`.
