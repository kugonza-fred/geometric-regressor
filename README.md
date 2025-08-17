
# **Geometric Regressor Documentation**
---
## **1. Description**
The **Geometric Regressor** is a supervised learning model that interprets input and output variables as points on a 2D coordinate plane. Each **input data point** (on the x-axis) forms a triangle with the **origin** \((0,0)\) and the **output data point** (on the y-axis). By measuring **geometric parameters** (distances, angles, areas, etc.) from these triangles, the model derives relationships that can be used to predict unknown outputs. This approach offers an **alternative perspective** to conventional regression techniques by leveraging **geometry-based features**.

---

## **2. Geometric Perspective**

### **2.1 2D Coordinate Plane Setup**
- **Origin (\(O\))**: \((0,0)\).  
- **Input Data Point (\(Q\))**: Placed on the **x-axis** at coordinate \((Q, 0)\).  
- **Output Data Point (\(T\))**: Placed on the **y-axis** at \((0, T)\).

Hence, for a single row \(i\) in the dataset, if \(Q_i\) is the input value and \(T_i\) is the target, we visualize the **triangle** \(Q_i O T_i\).

### **2.2 Geometric Parameters**
For each input point \(Q_i\) and its corresponding output \(T_i\):

1. **Angle** \(\theta_i\): The angle that the line \(\overline{Q_i T_i}\) makes with the x-axis.  
   \[
   \theta_i \;=\; \arctan\!\Bigl(\frac{T_i}{Q_i}\Bigr) \quad (\text{if } Q_i \neq 0).
   \]

2. **Distances**  
   - $\(\overline{Q_i O} = |Q_i|\)$.  
   - $\(\overline{T_i O} = |T_i|\)$.  
   - $\(\overline{Q_i T_i} = \sqrt{Q_i^2 + T_i^2}\)$.

3. **Distance Ratios**  
   - Example: $\(\dfrac{\overline{Q_i T_i}}{\overline{Q_i O}} = \dfrac{\sqrt{Q_i^2 + T_i^2}}{|Q_i|}\)$.  
   - These ratios normalise distances, making them **less sensitive** to scale.

4. **Area-Perimeter Ratios**  
   - **Area** of triangle \(Q_i O T_i\): \(\dfrac{1}{2}\,|Q_i|\,|T_i|\).  
   - **Perimeter**: \(|Q_i| + |T_i| + \sqrt{Q_i^2 + T_i^2}\).  
   - A ratio like \(\dfrac{\text{Area}}{\text{Perimeter}} = \dfrac{\frac{1}{2} |Q_i| |T_i|}{|Q_i| + |T_i| + \sqrt{Q_i^2 + T_i^2}}\) provides a **dimensionless** feature capturing shape characteristics.

These parameters can be **aggregated** (e.g., averaged) across training rows for each input column to build **column-specific relationships** with the target.

---

## **3. Preprocessing**

### **3.1 Outlier Detection Options**
The **Geometric Regressor** class provides several strategies for **detecting** outliers:

1. **Z-score**: Identifies points beyond a chosen threshold (e.g., \(\pm 3\) standard deviations).  
2. **Isolation Forest**: A tree-based model that isolates outliers by random splits.  
3. **DBSCAN**: A density-based clustering approach that flags points not belonging to dense clusters.  
4. **Ignore Outliers**: The user can opt to **bypass** outlier detection if they’ve handled them externally or wish to keep them.

### **3.2 Outlier Handling Options**
After detection, users can choose how to **handle** outliers:

1. **Capping** (Winsorizing): Replace extreme values with the nearest acceptable boundary.  
2. **Transformation**: Apply transformations (e.g., log, sqrt) to reduce outlier impact.  
3. **Removal**: Remove outlier rows entirely from the dataset.  
4. **No Handling**: Keep the outliers as is.

### **3.3 Integration with the Geometric Regressor Class**
All these **outlier** options can be passed as **instance variables** or **constructor parameters** to the **GeometricRegressor** class, allowing users to tailor preprocessing.

---

## **4. Training the Model**

### **4.1 Correlation-Based Weights**
- For each **input column** \(X\) and the **output** column \(T\), compute the **correlation coefficient** \(\rho_{X,T}\).  
- Define a weight \(w_X\) (initially) as the absolute value of this correlation, normalised so that all weights sum to 1 (or another convenient scale).

\[
w_X = 
\frac{|\rho_{X,T}|}
{\sum_{Y \in \{\text{all columns}\}} 
|\rho_{Y,T}|}.
\]

### **4.2 Calculating the Geometric Parameters**
Within each column \(X\):

1. For every row \(i\), compute:
   - **Angle** \(\theta_{X,i}\).  
   - **Distance Ratios** \(\dfrac{\sqrt{X_i^2 + T_i^2}}{|X_i|}\) (if \(X_i \neq 0\)).  
   - **Area-Perimeter** ratio, etc.
2. (Optionally) **aggregate** these parameters across all rows in the training set (e.g., take the mean angle, mean ratio).  
3. Store or learn **column-specific** references (e.g., \(\bar{\theta}_X\), \(\overline{\text{ratio}}_X\)).

---

## **5. Validation & Prediction**

Given a **new row** with input \((A_{\text{new}}, B_{\text{new}}, \ldots)\), the Geometric Regressor proceeds as follows:

1. **Compute Geometric Parameters** for each column \(X\). For instance, for column \(A\):
   - \(\theta_{A,\text{new}}\), distance ratio \(\dfrac{\sqrt{A_{\text{new}}^2 + T^2}}{|A_{\text{new}}|}\), etc.  
   - In practice, \(T\) is unknown at prediction time; you can **approximate** or rely on **training-set statistics** (like the mean angle).  
   - Alternatively, you may have a direct formula or iterative method for obtaining a partial prediction.

2. **Generate Predictions** for Each Parameter  
   - If each parameter yields a **slightly different** predicted \(T\), you might take the **mean** of those predictions within the same column.  
   - Let \(\hat{T}_{A}\) be the final predicted value from column \(A\). Similarly, compute \(\hat{T}_{B}\), \(\hat{T}_{C}\), etc.

3. **Weighted Aggregation**  
   - Combine the **column-level** predictions using the previously computed **weights** \(w_A, w_B, \ldots\).  
   \[
   \hat{T}_{\text{final}} 
   = 
   \frac{w_A \,\hat{T}_{A} \;+\; w_B \,\hat{T}_{B} \;+\; \cdots}
        {w_A + w_B + \cdots}.
   \]
   - The user may choose **mean**, **median**, or **trimmed mean** across columns if they prefer a different aggregation:
     \[
     \hat{T}_{\text{final}} 
     = 
     \text{median}\bigl(\hat{T}_{A}, \hat{T}_{B}, \ldots\bigr).
     \]

---

## **6. Model Variables & User Options**

1. **Outlier Detection**: Z-score, Isolation Forest, DBSCAN, or **ignore**.  
2. **Outlier Handling**: Capping, Transformation, Removal, or **none**.  
3. **Aggregation Method**: Weighted mean (default), median, or trimmed mean of the predictions.  
4. **Weights**:  
   - Correlation-based (default).  
   - Other weighting schemes (future versions might allow custom functions).  
5. **Parameter Calculation**: Users can **enable/disable** certain geometric parameters (e.g., only use angle and distance ratio, skip area-perimeter).  
6. **Other**:  
   - **Random seed** for reproducible outlier detection (Isolation Forest/DBSCAN).  
   - **Verbose** or **silent** mode for logging.

---

## **7. Potential Improvements & Future Directions**

1. **Adding a Z-axis**:  
   - Extending the geometric perspective to **3D** might capture more complex relationships or additional features.  
   - Example: Mapping a secondary dependent variable or a derived feature onto a third axis.

2. **Normalizing Weights**:  
   - You can **force** all weights to lie in \([0,1]\) by normalizing after each update or iteration.

3. **Cross-Validation for Weight Tuning**:  
   - Instead of relying on correlation alone, **k-fold cross-validation** could refine or optimize the weights for better generalization.

4. **Alternate Methods for Weighting**:  
   - One idea is to **fit a linear regression** on the per-column predictions (stacking approach) and use those coefficients as final weights.

5. **Iterative/Refinement Approaches**:  
   - If the geometric parameters depend on unknown \(T\) at prediction time, an **iterative** approach could refine \(T\) until convergence.

6. **Robust Statistics**:  
   - Incorporating **median** or **trimmed** mean at each stage to reduce the impact of outliers.

---

## **8. Additional Sections (Optional)**

- **Implementation Details**: Outline the structure of the **GeometricRegressor** class, method signatures, and example usage.  
- **Examples & Tutorials**: Provide **code snippets** (e.g., Python) demonstrating how to instantiate the class, pass in outlier detection/handling methods, train, and predict.  
- **Performance Metrics**: Suggest measuring **MSE**, **MAE**, or **\(R^2\)** to evaluate model performance on a validation/test set.  

---

