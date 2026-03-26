# 📊 t-SNE — t-Distributed Stochastic Neighbor Embedding

> A hands-on, visual guide to understanding and applying t-SNE for dimensionality reduction using Python and scikit-learn.

---

## 📖 Overview

This notebook provides a comprehensive walkthrough of **t-SNE (t-Distributed Stochastic Neighbor Embedding)** — one of the most powerful non-linear dimensionality reduction techniques used in data science and machine learning. Through step-by-step code examples and vivid visualizations, you will learn:

- **Why** t-SNE uses the t-distribution instead of the Gaussian in low-dimensional space
- **How** to apply t-SNE on real-world datasets (Iris, Digits)
- **How** perplexity affects the output structure
- **How** t-SNE compares to PCA on linear vs. non-linear data
- **How** to run t-SNE in 3D

---

## 🧰 Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |
| `scikit-learn` | t-SNE, PCA, datasets |
| `scipy` | Statistical distributions |

---

## 📂 Notebook Structure

### Cell 1 — Imports & Setup
All necessary libraries are imported and global plot settings are configured.

---

### Cell 2 — Gaussian vs t-Distribution
Illustrates **why t-SNE uses the t-distribution** in low-dimensional space. The t-distribution's heavier tails prevent the "crowding problem" — points that are moderately distant in high dimensions can still be well-separated in 2D.

<img width="904" height="489" alt="Screenshot 2026-03-27 032000" src="https://github.com/user-attachments/assets/44af4f0f-a003-4025-bcee-e95613d86e1f" />


> **Key insight:** The red shaded area shows the heavier tail of the t-distribution, which allows moderately dissimilar points to have more room in the low-dimensional embedding.

---

### Cell 3 — t-SNE vs PCA on the Iris Dataset
Applies both **PCA (linear)** and **t-SNE (non-linear)** to the classic Iris dataset (150 samples, 4 features, 3 classes) and compares the 2D projections side by side.

<img width="1411" height="646" alt="Screenshot 2026-03-27 032010" src="https://github.com/user-attachments/assets/d71ec354-2c56-44f0-970f-a94f25499ac4" />


> **Key insight:** t-SNE produces tighter, more separated clusters compared to PCA's linear projection. The `setosa` class is well separated by PCA, but `versicolor` and `virginica` overlap — t-SNE cleanly separates all three.

---

### Cell 4 — Digits Dataset Preview
Loads the **Digits dataset** (1797 samples, 64 features — 8×8 pixel images of handwritten digits 0–9) and displays sample images for each class.

<img width="1485" height="422" alt="Screenshot 2026-03-27 032019" src="https://github.com/user-attachments/assets/95083819-b193-47ba-ad3d-b3fd511612d8" />


---

### Cell 5 — t-SNE on Digits (PCA → t-SNE Pipeline)
Demonstrates the **best practice pipeline for large datasets**: first reduce dimensions with PCA (to 50 components), then apply t-SNE to 2D. This dramatically speeds up t-SNE while preserving structure.
```python
# Best practice pipeline
X_pca50 = PCA(n_components=50).fit_transform(X_scaled)
X_tsne  = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(X_pca50)
```

<img width="673" height="629" alt="Screenshot 2026-03-27 032040" src="https://github.com/user-attachments/assets/cfc1a7d8-73ff-482b-8f8b-f6dd9256c6db" />


> **Key insight:** t-SNE clearly separates all 10 digit classes into distinct clusters, with digit labels annotated at each cluster center. Notice how similar-looking digits (e.g., 3 & 8, 4 & 9) are positioned closer together.

---

### Cell 6 — Effect of Perplexity
Explores how the **perplexity hyperparameter** (range: 5–50, default: 30) controls the trade-off between local and global structure.

<img width="1207" height="370" alt="Screenshot 2026-03-27 032054" src="https://github.com/user-attachments/assets/102a40db-5c26-4732-af22-0f2d0f112ae2" />


| Perplexity | Behavior |
|---|---|
| **5** | Very tight clusters — fragmented, noisy |
| **15** | Moderate local focus |
| **30** ✅ | Default — best balance for most datasets |
| **50** | Clusters spread out — more global structure visible |

> **Key insight:** Perplexity roughly corresponds to the number of effective nearest neighbors. Too low → clusters fragment; too high → structure blurs.

---

### Cell 7 — Swiss Roll: PCA vs t-SNE
Uses the **Swiss Roll** manifold dataset to clearly demonstrate t-SNE's power over PCA on non-linear data.

<img width="1537" height="544" alt="Screenshot 2026-03-27 032116" src="https://github.com/user-attachments/assets/549c050d-819c-44a0-bac6-427e96875ffc" />


> **Key insight:** PCA fails to "unroll" the manifold because it can only capture linear relationships. t-SNE successfully unrolls the Swiss Roll by preserving local neighborhood structure — a hallmark of non-linear methods.

---

### Cell 8 — t-SNE in 3D
Applies t-SNE with `n_components=3` to the Iris dataset and renders a **3D scatter plot**.

<img width="694" height="692" alt="Screenshot 2026-03-27 032126" src="https://github.com/user-attachments/assets/7f35b2fc-b06e-457b-b7b9-bc651213f44a" />


> **Tip:** 3D t-SNE can reveal additional structure hidden in 2D projections, though it is harder to interpret visually. Use 2D for most practical purposes.

---

### Cell 9 — High-Dimensional Blobs: PCA vs t-SNE
Creates synthetic data with **8 clusters in 50 dimensions** (800 samples) and compares PCA vs t-SNE projections.
<img width="1562" height="752" alt="image" src="https://github.com/user-attachments/assets/dff36bdf-6da5-4baa-a3a4-036c45bab428" />

> **Key insight:** In high dimensions, t-SNE produces far cleaner cluster separation than PCA. PCA's clusters overlap significantly, while t-SNE yields clearly distinct groups.

---

## 📌 Key Takeaways

| Aspect | t-SNE | PCA |
|---|---|---|
| Type | Non-linear | Linear |
| Goal | Preserve local structure | Maximize global variance |
| Scalability | Slow on large data (use PCA first) | Fast |
| Interpretability | Axes have no meaning | Axes = principal components |
| Best for | Visualization (2D/3D) | Preprocessing, compression |
| Crowding problem | Solved via t-distribution | Not applicable |

---

## 💡 Best Practices

1. **Always standardize** your features before running t-SNE.
2. **Use PCA first** on large or high-dimensional datasets (`n_components=50`) to speed up t-SNE.
3. **Tune perplexity** in the 5–50 range; try multiple values.
4. **Do not interpret axes** — t-SNE axes have no inherent meaning.
5. **Run multiple times** — t-SNE is stochastic; fix `random_state` for reproducibility.
6. **Use `learning_rate='auto'` and `init='pca'`** for better convergence (scikit-learn ≥ 1.2).

---

## 📚 References

- [Original t-SNE Paper — van der Maaten & Hinton (2008)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- [scikit-learn TSNE Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [How to Use t-SNE Effectively — Wattenberg et al.](https://distill.pub/2016/misread-tsne/)

---
