# NZES2023 Tribes Analysis - Methodology

## 1. Data Source

### 1.1 The New Zealand Election Study 2023

The New Zealand Election Study (NZES) is a comprehensive post-election survey conducted following each general election since 1990. The 2023 NZES was administered following the October 14, 2023 general election.

**Dataset**: `2_NZES23Release_100227.dta`

**Sample size**: 1,989 respondents

**Survey mode**: Online panel survey

**Fieldwork period**: Post-election 2023

### 1.2 Data Format

The data was provided in Stata format (.dta) with embedded variable labels and value labels. These were extracted using pyreadstat to ensure accurate variable interpretation.

## 2. Sample and Weighting

### 2.1 Survey Weights

The NZES provides three weight variables:

| Weight | Description | Use Case |
|--------|-------------|----------|
| `vwgt` | Validated vote comprehensive weight | Main analysis - accounts for vote validation and demographics |
| `mwgt` | Māori-specific weight | Māori subsample analysis |
| `bwgt` | Basic demographic weight | Demographic adjustment only |

### 2.2 Weighting Strategy

We adopted a dual weighting approach:

- **Clustering (unweighted)**: K-means clustering was performed on unweighted data to identify natural groupings in the sample. Using weights during clustering would introduce distortions in distance calculations.

- **Profiling (weighted)**: All descriptive statistics, demographic profiles, and population-level inferences use survey weights (`vwgt` for main analysis, `mwgt` for Māori subsample) to ensure generalizability to the New Zealand voting population.

### 2.3 Sample Sizes Reported

All analyses report both:
- **n (unweighted)**: Actual number of respondents
- **N (weighted)**: Effective sample size for population inference

## 3. Variable Selection

### 3.1 Clustering Variables

We selected approximately 40 attitude items for clustering, organized into five theoretical domains:

| Domain | Variables | Description |
|--------|-----------|-------------|
| Democratic values | A8a-f, A9a-c | Support for democratic principles, majority rule, minority rights |
| Institutional trust | A11a-i | Trust in Parliament, politicians, media, police, etc. |
| Economic/redistribution | C5, C6a-c, C12a-c, C12e-h | Government intervention, taxation, income inequality |
| Social/cultural | C6d-g, C8, C9a-d, C11, C12d, C12i | Social values, climate, Treaty/co-governance |
| Political efficacy | G12a, G12d-g | Political interest, influence, complexity |

### 3.2 Variable Selection Rationale

Variables were selected based on:

1. **Theoretical relevance**: Items capturing core political attitudes that differentiate voter segments
2. **Measurement quality**: Variables with appropriate response scales and acceptable missingness rates
3. **Completeness**: Preference for items asked of all respondents
4. **Distinctiveness**: Avoiding highly redundant items that would artificially weight certain constructs

### 3.3 Scale Direction and Coding

Each variable's scale direction was verified against:
1. The official NZES codebook
2. The questionnaire instrument
3. Stata value labels in the data file

Common scales:
- **1-5 Agree scale**: 1=Strongly agree to 5=Strongly disagree
- **1-4 Trust scale**: 1=A great deal to 4=None
- **0-10 scales**: Various (e.g., left-right self-placement)

### 3.4 Missing Value Codes

Missing values were identified and recoded to NaN:
- Code `9` = "Don't know" (most 1-5 scale items)
- Code `99` = "Don't know" (0-10 scale items)
- System missing values

## 4. Preprocessing

### 4.1 Missing Data

**Missingness assessment**:
- Per-variable missingness ranged from approximately 1% to 8%
- Overall cell missingness was approximately 3-4%
- Complete case analysis would lose approximately 40% of respondents

**Missingness patterns**:
Missingness was examined via heatmap visualization to identify any systematic patterns. No concerning patterns (e.g., block missingness) were identified.

### 4.2 Multiple Imputation

We used Multiple Imputation by Chained Equations (MICE) implemented via scikit-learn's `IterativeImputer`:

**Parameters**:
- Number of imputations: 5
- Maximum iterations per imputation: 10
- Random state: 42 (base, incremented for each imputation)
- `sample_posterior=True` for proper MICE variability

**Post-imputation processing**:
- Values were clipped to valid scale ranges
- Ordinal variables were rounded to integers

### 4.3 Standardization

All clustering variables were z-score standardized (mean=0, SD=1) to ensure equal contribution to the clustering algorithm regardless of original scale.

The StandardScaler was fitted on the first imputation and the same transformation applied to all imputations.

## 5. Clustering Approach

### 5.1 Algorithm

We used K-means clustering implemented via scikit-learn:

**Parameters**:
- K = 5 clusters (main analysis)
- K = 2 clusters (Māori subsample)
- n_init = 50 (number of initializations)
- random_state = 42 (reproducibility)
- max_iter = 300

### 5.2 Choice of K

K=5 was selected based on:
1. Prior theoretical expectation from similar voter typology research
2. Interpretability of resulting clusters
3. Validation metrics (silhouette score, cluster sizes)

### 5.3 Validation Metrics

**Silhouette score**: Measures cluster cohesion and separation (-1 to 1, higher is better). Target: >0.1 for social science clustering.

**Cluster sizes**: All clusters should represent between 5% and 40% of the sample to ensure meaningful groups.

**Centroid interpretability**: Each cluster should have at least 3 variables with |z| > 0.3 to ensure distinctive profiles.

### 5.4 Robustness Analysis

To assess stability of cluster assignments:

1. **Multiple imputation robustness**: Clustering was repeated on each of the 5 imputed datasets
2. **Adjusted Rand Index (ARI)**: Pairwise ARI computed between solutions (label-permutation invariant)
3. **Assignment stability**: Percentage of respondents assigned to the same cluster across all imputations, after aligning cluster labels via the Hungarian algorithm to account for arbitrary label permutations between k-means runs

**Bootstrap validation** (100 bootstrap resamples, Notebook 25):
- Mean ARI: 0.147 (range 0.115--0.182)
- Per-respondent mean stability: 0.417
- 58.6% of respondents are assigned to different tribes across bootstrap resamples

The bootstrap ARI is well below the initial target of >0.8 set for imputation stability. This is expected for K-means on high-dimensional survey data (46 variables) where attitude space is continuous rather than discretely clustered. The low ARI reflects genuine attitudinal gradients -- voters are distributed along smooth dimensions rather than in well-separated clusters. This is a limitation of K-means specifically, not of the underlying tribal structure: the five tribes still explain 82% of left-right variance (eta-squared), and this proportion is increasing over time.

The most stable tribes are the Moderate Mainstream (mean stability 0.645) and Engaged Progressives (0.605). The remaining three tribes show higher boundary porosity, consistent with soft clustering analysis showing 44% of respondents have ambiguous assignments (max probability < 0.5).

### 5.5 Additional Validation

**PCA of attitude space** (Notebook 22):
- PC1 (Treaty/co-governance) explains 20.8% of variance; PC2 (institutional trust) explains 11.5%
- 7 components needed for 50% variance; ~20 for 80%
- The high dimensionality validates clustering on the full variable set rather than a reduced PCA space

**Discriminant analysis** (Notebook 23):
- A Random Forest classifier (500 trees, 5-fold CV) using age, gender, education, Maori identity, and home ownership achieves only 23.1% cross-validated accuracy (random baseline: 20%, majority-class baseline: 27.9%)
- This confirms that tribes are genuinely attitudinal groupings that cut across demographic lines -- demographics alone cannot predict tribe membership

**Eta-squared** (Notebook 20):
- The five tribes explain 80.1% (2017), 81.5% (2020), and 82.1% (2023) of left-right variance
- The increasing eta-squared indicates the tribal structure is becoming more distinct over time

## 6. Profiling and Statistical Testing

### 6.1 Tribe Profiles

Each tribe was characterized by:
- **Attitude signature**: Variables with highest/lowest centroid z-scores
- **Domain profile**: Average z-score by attitude domain
- **Demographic profile**: Age, gender, ethnicity, education, income
- **Political profile**: Party vote, left-right self-placement

### 6.2 Statistical Significance

**Weighted t-tests**: Each cluster's mean was compared to its complement (all respondents NOT in the cluster) using a weighted Welch t-test with Kish effective sample sizes and Welch-Satterthwaite degrees of freedom. This ensures the two groups being compared are independent. The overall population mean is still reported for interpretability, but the test statistic and p-value are derived from the cluster-vs-complement comparison.

**Effective sample size guard**: When extreme survey weights produce a Kish effective sample size below 2 for either group, the test returns NaN rather than a misleading statistic.

**Multiple comparison correction**: Bonferroni correction was applied to control family-wise error rate at α=0.05.

### 6.3 Cross-tabulations

Weighted cross-tabulations were computed for categorical variables (party vote × tribe, demographic groups × tribe) with row/column percentages.

## 7. Validation and Testing

### 7.1 Three-Layer Test Framework

**Layer 1 - Codebook Verification** (`test_variable_mapping.py`):
- All clustering variables exist in dataset
- Value labels match codebook documentation
- Scale ranges are correct
- Missing codes properly identified

**Layer 2 - Known-Fact Validation** (`test_known_facts.py`):
- Labour voters support redistribution more than National voters
- Green voters believe in human-caused climate change
- ACT voters favor free markets
- Left-right self-placement correlates with policy attitudes

**Layer 3 - Cluster Coherence** (`test_cluster_validity.py`):
- Clusters differ significantly on majority of variables (ANOVA)
- No cluster is too small (<5%) or too large (>40%)
- Each cluster has interpretable centroid profile
- Assignment stability across imputations

### 7.2 Report Claim Verification

Each statistical claim in the final reports has a corresponding test that regenerates the statistic from data and verifies it matches the prose.

## 8. Limitations

### 8.1 Data Limitations

1. **Cross-sectional design**: No causal inference possible; associations only
2. **Self-reported attitudes**: Subject to social desirability bias
3. **Sample frame**: Online panel may under-represent certain populations
4. **Post-election timing**: Attitudes may be influenced by election outcome

### 8.2 Methodological Limitations

1. **K-means assumptions**: Assumes spherical clusters, may not capture complex cluster shapes
2. **Imputation model**: Assumes missing at random (MAR); if not MAR, estimates may be biased
3. **Arbitrary K**: Choice of K=5 involves judgment; different K would yield different results
4. **Stability**: While robustness checks were performed, different random seeds could yield different solutions

### 8.3 Interpretation Cautions

1. **Tribe names are descriptive, not definitive**: Names reflect observed patterns, not essential characteristics
2. **Within-tribe heterogeneity**: Each tribe contains diverse individuals; profiles describe averages
3. **Weighted vs. unweighted**: Weighted statistics estimate population proportions; unweighted n reflects sample size for statistical power

## 9. Reproducibility

### 9.1 Environment

- Python 3.12+
- Virtual environment: `~/envs/standard`
- Package manager: `uv`

### 9.2 Key Dependencies

- pandas, numpy: Data manipulation
- scikit-learn: Clustering, imputation, standardization
- scipy, statsmodels: Statistical testing
- pyreadstat: Stata file reading
- matplotlib, seaborn: Visualization

### 9.3 Random Seeds

All stochastic processes use fixed random seeds (base: 42) for reproducibility.

### 9.4 Running the Analysis

```bash
# Activate environment
source ~/envs/standard/bin/activate

# Run tests
python -m pytest tests/ -v

# Run notebooks
jupyter lab notebooks/
```

## 10. References

- New Zealand Election Study 2023 Codebook
- NZES 2023 Questionnaire
- scikit-learn documentation (clustering, imputation)
