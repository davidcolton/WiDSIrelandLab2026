# WiDS 2026 Wildfire Prediction - Improvement Plan

## Executive Summary

Based on a comprehensive comparison with top-performing solutions (particularly **wids-n20** which achieved LB ~0.97167), this document outlines a strategic improvement plan for our wildfire evacuation zone impact prediction model.

**Current Status**: Strong foundation with excellent EDA, documentation, and code quality (78/110 score)  
**Target**: Implement high-impact improvements from top performers to boost predictive performance

---

## Assessment of Suggested Improvements

### ✅ **AGREE** - High-Impact Improvements to Implement

#### 1. Replace RSF with GradientBoostingSurvivalAnalysis (GBSA)
**Priority**: 🔴 CRITICAL  
**Expected Impact**: +0.10-0.15 C-index improvement  
**Rationale**: 
- GBSA consistently outperforms RSF on tabular survival data
- Better handles non-linear relationships and feature interactions
- More efficient training and prediction
- Already imported in notebook 04 but not used

**Implementation**:
- Train GBSA as primary model instead of RSF
- Use multiple configurations (5 variants with different hyperparameters)
- Keep RSF for ensemble comparison

---

#### 2. Add 5-Fold StratifiedKFold with Multiple Seeds
**Priority**: 🔴 CRITICAL  
**Expected Impact**: Biggest score gain through variance reduction  
**Rationale**:
- Current single 80/20 split has high variance
- wids-n20 uses 5-fold × 20 seeds = 100 models per config
- Dramatically improves prediction stability
- Standard practice for competitions

**Implementation**:
- Implement 5-fold StratifiedKFold (stratify on `event`)
- Use 5-10 seeds (balance between performance and compute time)
- Average predictions across all folds and seeds
- Generate out-of-fold (OOF) predictions for validation

---

#### 3. Add Effective ETA Feature
**Priority**: 🟡 HIGH  
**Expected Impact**: Captures critical time-to-impact signal  
**Rationale**:
- Combines two threat signals into interpretable physical quantity
- Formula: `effective_eta = dist / (closing_speed + radial_growth)`
- Accounts for both fire movement AND expansion
- Missing from current feature set

**Implementation**:
```python
# In notebook 03_feature_engineering.ipynb
X_train['effective_eta'] = X_train['dist_min_ci_0_5h'] / (
    X_train['closing_speed_m_per_h'].abs() + 
    X_train['radial_growth_rate_m_per_h'] + 
    1  # Avoid division by zero
)
```

---

#### 4. Add Zone Flag Features
**Priority**: 🟡 HIGH  
**Expected Impact**: Enables regime-specific modeling  
**Rationale**:
- Fires behave differently at different distances
- Allows model to learn distance-dependent patterns
- Aligns with 5km competition threshold

**Implementation**:
```python
# Critical zone: < 5km (within competition threshold)
X_train['zone_critical'] = (X_train['dist_min_ci_0_5h'] < 5000).astype(int)

# Warning zone: 5-15km (approaching)
X_train['zone_warning'] = (
    (X_train['dist_min_ci_0_5h'] >= 5000) & 
    (X_train['dist_min_ci_0_5h'] < 15000)
).astype(int)

# Safe zone: >= 15km (distant)
X_train['zone_safe'] = (X_train['dist_min_ci_0_5h'] >= 15000).astype(int)
```

---

#### 5. Hard-Assign prob_72h for Near/Far Zones
**Priority**: 🟡 HIGH  
**Expected Impact**: Exploits structural rule, reduces noise  
**Rationale**:
- **Logical certainty**: If fire is within 5km at t0+5h, it will almost certainly be within 5km at 72h
- **Structural rule**: Competition defines "hit" as coming within 5km
- **Noise reduction**: Removes model uncertainty for obvious cases

**Implementation**:
```python
# After generating predictions
near_zone_mask = X_test['dist_min_ci_0_5h'] < 5000
far_zone_mask = X_test['dist_min_ci_0_5h'] > 50000  # Very distant fires

# Hard assignments
submission.loc[near_zone_mask, 'prob_72h'] = 1.0  # Near fires will stay near
submission.loc[far_zone_mask, 'prob_72h'] = 0.0   # Far fires won't reach
```

---

#### 6. Add IPCW-Weighted LightGBM Classifier for 24h/48h
**Priority**: 🟠 MEDIUM  
**Expected Impact**: Better calibration at specific horizons  
**Rationale**:
- GBSA predicts survival curves; LGB predicts specific horizons
- IPCW (Inverse Probability of Censoring Weighting) handles censored data properly
- Different models excel at different time horizons
- Ensemble GBSA + LGB for 24h and 48h

**Implementation**:
- Create binary targets for 24h and 48h horizons
- Compute IPCW weights to handle censoring
- Train separate LGB classifiers for each horizon
- Blend with GBSA predictions

---

#### 7. Implement Regime-Conditioned Calibration
**Priority**: 🟠 MEDIUM  
**Expected Impact**: Improved probability calibration  
**Rationale**:
- Different fire regimes need different calibration
- Near-zone stable fires vs. near-zone sparse fires behave differently
- Calibration floors prevent over-confident predictions

**Implementation**:
```python
# Define regimes
near_stable = (dist < 5000) & (growth_rate > threshold)
near_sparse = (dist < 5000) & (growth_rate <= threshold)
far = (dist >= 5000)

# Apply regime-specific calibration
for regime in [near_stable, near_sparse, far]:
    # Apply isotonic regression or Platt scaling per regime
    calibrated_probs = calibrate_predictions(raw_probs[regime], regime_type)
```

---

#### 8. Add Optuna Hyperparameter Search
**Priority**: 🟢 LOW (but valuable)  
**Expected Impact**: 5-10% performance improvement  
**Rationale**:
- Current models use default or hand-tuned parameters
- Systematic search finds better configurations
- Optimize for competition metric (Hybrid Score)

**Implementation**:
```python
import optuna

def objective(trial):
    # Define hyperparameter search space
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }
    
    # Train model and return validation score
    model = GradientBoostingSurvivalAnalysis(**params)
    # ... train and evaluate
    return validation_c_index

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

### ❌ **DISAGREE** - Not Recommended

None of the suggestions are fundamentally flawed. All are valid improvements. However, we prioritize based on:
- **Implementation complexity** vs. **expected impact**
- **Computational cost** vs. **score improvement**
- **Time constraints** for competition deadline

---

## Implementation Roadmap

### Phase 1: Critical Improvements (Week 1)
**Goal**: Implement highest-impact changes

1. **Replace RSF with GBSA** (Day 1-2)
   - Modify notebook 04
   - Train 5 GBSA configurations
   - Compare with RSF baseline

2. **Implement 5-Fold CV with Multiple Seeds** (Day 3-4)
   - Add cross-validation framework
   - Generate OOF predictions
   - Average across folds and seeds

3. **Add Effective ETA and Zone Flags** (Day 5)
   - Modify notebook 03
   - Add 4 new features
   - Regenerate engineered datasets

### Phase 2: Advanced Improvements (Week 2)
**Goal**: Add sophisticated techniques

4. **Hard-Assign prob_72h** (Day 6)
   - Implement near/far zone logic
   - Apply after prediction generation

5. **Add IPCW-Weighted LGB** (Day 7-9)
   - Implement IPCW calculation
   - Train LGB classifiers for 24h/48h
   - Blend with GBSA predictions

6. **Regime-Conditioned Calibration** (Day 10-11)
   - Define fire regimes
   - Implement per-regime calibration
   - Validate improvements

### Phase 3: Optimization (Week 3)
**Goal**: Fine-tune for maximum performance

7. **Optuna Hyperparameter Search** (Day 12-14)
   - Set up Optuna study
   - Run 100+ trials
   - Select best configurations

---

## Expected Performance Gains

| Improvement | Expected C-index Gain | Cumulative Gain |
|-------------|----------------------|-----------------|
| Baseline (RSF) | 0.750 | - |
| + GBSA | +0.030 | 0.780 |
| + 5-Fold CV | +0.020 | 0.800 |
| + New Features | +0.015 | 0.815 |
| + Hard Assignments | +0.010 | 0.825 |
| + LGB Ensemble | +0.015 | 0.840 |
| + Calibration | +0.010 | 0.850 |
| + Optuna | +0.010 | 0.860 |

**Target**: C-index > 0.85, Hybrid Score > 0.90

---

## File Structure Changes

### New Files to Create

```
ai_approach/
├── notebooks/
│   ├── 04_modelling_and_evaluation_v2.ipynb  # Updated with improvements
│   └── 05_advanced_ensemble.ipynb            # IPCW-LGB + calibration
├── src/                                       # NEW: Reusable code
│   ├── __init__.py
│   ├── cross_validation.py                   # CV framework
│   ├── feature_engineering.py                # Feature functions
│   ├── calibration.py                        # Calibration methods
│   └── ensemble.py                           # Ensemble logic
└── docs/
    ├── improvement_plan.md                   # This file
    └── implementation_notes.md               # Track progress
```

---

## Risk Mitigation

### Potential Issues

1. **Computational Cost**
   - 5-fold × 10 seeds = 50 models per config
   - **Mitigation**: Use parallel processing, start with fewer seeds

2. **Overfitting Risk**
   - More complex models may overfit
   - **Mitigation**: Strong CV framework, monitor train/val gap

3. **Implementation Bugs**
   - Complex ensemble logic prone to errors
   - **Mitigation**: Unit tests, validation checks, gradual rollout

4. **Time Constraints**
   - Full implementation may take 3 weeks
   - **Mitigation**: Prioritize Phase 1, defer Phase 3 if needed

---

## Success Metrics

### Validation Metrics
- **C-index**: > 0.85 (currently ~0.75)
- **Brier Score**: < 0.15 (lower is better)
- **Hybrid Score**: > 0.90 (competition metric)

### Code Quality Metrics
- **Documentation**: Maintain current high standard
- **Reproducibility**: All results reproducible with seed
- **Test Coverage**: Unit tests for critical functions

### Competition Metrics
- **Leaderboard Position**: Top 25% → Top 10%
- **Score Improvement**: +0.10 Hybrid Score points

---

## Conclusion

The suggested improvements are **well-founded and high-impact**. The wids-n20 solution demonstrates that these techniques work effectively for this competition.

**Recommendation**: Implement all 8 improvements in priority order, focusing on Phase 1 (critical improvements) first. This balanced approach maximizes score improvement while maintaining our strong foundation in EDA, documentation, and code quality.

**Key Insight**: Our current solution excels at **understanding the problem** (EDA, visualization, documentation). The improvements focus on **solving the problem better** (advanced modeling, ensembling, calibration). This combination positions us for top-tier performance.

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-26  
**Author**: AI Planning Mode  
**Status**: Ready for Implementation