# Titanic Data Pipeline Documentation

## Pipeline Overview
This pipeline preprocesses the Titanic dataset to prepare it for machine learning modeling. It handles categorical encoding, target encoding, outlier detection and treatment, feature scaling, and missing value imputation.

<img src="https://github.com/fickas/asynch_models/blob/main/transform_pipeline.png?raw=true" width="70%" alt="Pipeline Diagram">

## Step-by-Step Design Choices

### 1. Gender Mapping (`map_gender`)
- **Transformer:** `CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})`
- **Design Choice:** Binary encoding of gender with female as 1 and male as 0
- **Rationale:** Simple categorical mapping that preserves the binary nature of the feature without increasing dimensionality

### 2. Class Mapping (`map_class`)
- **Transformer:** `CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})`
- **Design Choice:** Ordinal encoding of passenger class from lowest (Crew) to highest (C1)
- **Rationale:** Preserves the inherent ordering of passenger classes on the Titanic

### 3. Target Encoding for Joined Column (`target_joined`)
- **Transformer:** `CustomTargetTransformer(col='Joined', smoothing=10)`
- **Design Choice:** Target encoding with smoothing factor of 10
- **Rationale:** 
  - Replaces the categorical 'Joined' feature with its relationship to the target variable
  - Smoothing=10 balances between using the global mean (high smoothing) and the category mean (low smoothing)
  - Helps address potential overfitting from rare categories

### 4. Outlier Treatment for Age (`tukey_age`)
- **Transformer:** `CustomTukeyTransformer(target_column='Age', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:** 
  - Outer fence (Q1-3×IQR, Q3+3×IQR) identifies only the most extreme outliers
  - Age may have legitimate outliers (very young or old passengers) that should be preserved unless extreme

### 5. Outlier Treatment for Fare (`tukey_fare`)
- **Transformer:** `CustomTukeyTransformer(target_column='Fare', fence='outer')`
- **Design Choice:** Tukey method with outer fence for identifying extreme outliers
- **Rationale:**
  - Fare prices have high variability and legitimate outliers for luxury accommodations
  - Outer fence preserves most of the original distribution while handling extreme values

### 6. Age Scaling (`scale_age`)
- **Transformer:** `CustomRobustTransformer(target_column='Age')`
- **Design Choice:** Robust scaling for Age feature
- **Rationale:**
  - Robust to outliers compared to standard scaling
  - Uses median and interquartile range instead of mean and standard deviation
  - Appropriate for Age which may not follow normal distribution

### 7. Fare Scaling (`scale_fare`)
- **Transformer:** `CustomRobustTransformer(target_column='Fare')`
- **Design Choice:** Robust scaling for Fare feature
- **Rationale:**
  - Fare has high variability and skewed distribution
  - Robust scaling reduces influence of remaining outliers after Tukey treatment

### 8. Imputation (`impute`)
- **Transformer:** `CustomKNNTransformer(n_neighbors=5)`
- **Design Choice:** KNN imputation with 5 neighbors
- **Rationale:**
  - Uses relationships between features to estimate missing values
  - k=5 balances between too few neighbors (overfitting) and too many (underfitting)
  - More appropriate than simple mean/median imputation given the relationships in Titanic data

## Pipeline Execution Order Rationale
1. Categorical encoding first to prepare for subsequent numerical operations
2. Target encoding next as it requires original categorical values
3. Outlier treatment before scaling to prevent outliers from affecting scaling parameters
4. Scaling before imputation so that distance metrics in KNN aren't skewed by unscaled features
5. Imputation last to fill missing values using all preprocessed features

## Performance Considerations
- RobustScaler instead of StandardScaler due to presence of outliers
- KNN imputation instead of simple imputation to preserve relationships between features
- Target encoding with smoothing for categorical features with many levels
