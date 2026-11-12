# 🍅 Rotten Tomatoes AI Sentiment Classifier

> Automated Movie Review Classification System using Logistic Regression and Gradient Descent

## Quick Intro

- [📝 Project Brief](./Project_Brief.pdf)

## Overview

Review platforms like **Rotten Tomatoes** need every review labeled positive (Fresh 🍅) or negative (Rotten 🤢). This project builds a **sentiment classifier from scratch** that reaches **73.3% validation accuracy** on 3,554 held-out reviews (F1 0.730).

Unlike black-box neural networks, this implementation provides **full transparency**—every word's weight is visible (e.g., "performances" → +1.47, "boring" → -1.62), enabling editorial teams to audit decisions and maintain stakeholder trust.

### Key Features

- **Accuracy**: 96.4% training accuracy, 73.3% validation accuracy
- **Interpretable**: Transparent word weights for audit trails
- **Fast**: Scores all 3,554 validation reviews in well under a second, in pure Python
- **Production-Ready**: Interactive CLI for real-time predictions
- **From-Scratch Implementation**: No ML frameworks—proves deep mathematical understanding

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Accuracy** | **96.4%** |
| **Validation Accuracy** | **73.3%** |
| **Validation F1** | **0.730** |
| **False positives vs. false negatives** | 481 vs. 469 |

These numbers come from `_eval/results.md`, which `python3 _eval/eval_metrics.py` regenerates (deterministic, about 40 seconds). `_eval/CLAIMS.md` maps every number in this README to its source.

### Model Insights

**Top Positive Words** (from `weights` file):
```
performances   +1.47
world          +1.36
moving         +1.34
heart          +1.29
works          +1.29
```

**Top Negative Words**:
```
bad            -1.75
boring         -1.62
dull           -1.61
feels          -1.50
silly          -1.30
```

### Example Prediction

**Input Review** (written for this demo, not from the dataset):
> "Submerge by Movements is a competent but uninspired film"

**Model Output** (`python3 interactive.py`):
- **Score** w·Φ(x): -0.26
- **Predicted**: Negative (-1)
- **Probability of positive** (sigmoid of the score): 43.5%

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Input: Movie Review                      │
│          "Heath Ledger gives one of the best..."            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction (Sparse Vectors)             │
│   extractWordFeatures() or extractCharacterFeatures(n)      │
│   Output: {'Heath': 1, 'Ledger': 1, 'best': 1, ...}        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Logistic Regression Model                   │
│         h = σ(w·Φ(x)) = 1/(1 + e^(-w·Φ(x)))                │
│              w = learned weight vector                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output: Classification                      │
│         Positive (1) or Negative (-1) + Confidence          │
└─────────────────────────────────────────────────────────────┘
```

### Core Algorithm

**Logistic Regression with Stochastic Gradient Descent**

```python
# Loss Function (Binary Cross-Entropy)
Loss(y, h) = -(y·log(h) + (1-y)·log(1-h))

# Prediction Function (Sigmoid)
h = σ(w·Φ(x)) = 1 / (1 + e^(-w·Φ(x)))

# Gradient Descent Update Rule
w ← w - α·(h-y)·Φ(x)

# Hyperparameters
α (learning rate) = 0.01
epochs = 40
```

### Why No ML Frameworks?

This project **deliberately avoids** scikit-learn, TensorFlow, and PyTorch to demonstrate:

**Mathematical understanding** - Hand-derived gradient formulas  
**Algorithm proficiency** - Implemented SGD from first principles  
**Memory efficiency** - Custom sparse vector operations (Dict[str, float])  
**Production skills** - Can optimize and debug at the algorithmic level  

---

## 🔬 Technical Deep Dive

### Milestone 1: Manual Gradient Descent

**Objective**: Understand optimization through hand calculation

Manually computed one pass of gradient descent (α = 0.5) over 4 mini-reviews:
1. "(0) pretty bad"
2. "(1) good plot"
3. "(0) not good"
4. "(1) pretty scenery"

Starting weights (pretty, good, bad, plot, not, scenery): `[0, 0, 0, 0, 0, 0]`  
Final weights: `[0.031088, -0.031088, -0.25, 0.25, -0.281088, 0.281088]`

**Key Learning**: Understanding why `w = w - α·(h-y)·Φ(x)` works intuitively

### Milestone 2: Mathematical Derivation

**Objective**: Prove gradient formula from first principles

Derived using chain rule:
```
∂Loss/∂w = (∂Loss/∂h) · (∂h/∂k) · (∂k/∂w)
         = (h-y) · Φ(x)
```

Where:
- `∂Loss/∂h = (1-y)/(1-h) - y/h`
- `∂h/∂k = h·(1-h)` (sigmoid derivative)
- `∂k/∂w = Φ(x)` (linear derivative)

**Key Learning**: Why gradient descent minimizes loss automatically

### Milestone 3: Sparse Vector Operations

**Challenge**: 11,622 training-set words × 3,554 training reviews = 41.3 million potential entries

**Solution**: Store only non-zero features using Python dictionaries

```python
# Dense representation (wasteful)
dense_vector = [0, 0, 0, ..., 1, 0, 0, 1, ...]  # 11,622 elements

# Sparse representation (efficient)
sparse_vector = {'good': 1, 'movie': 1, 'best': 1}  # ~19 elements on average
```

**Memory Reduction**: 99.84% (from ~41.3M to ~67K stored entries)

**Implemented Operations**:
- `extractWordFeatures(x)`: Text → Dict[str, int]
- `increment(d1, scale, d2)`: d1 += scale * d2
- `dotProduct(d1, d2)`: Compute inner product

### Milestone 4: Complete Training Pipeline

**Dataset**:
- Training: 3,554 reviews (polarity.train)
- Validation: 3,554 reviews (polarity.dev)

**Training Process**:
```python
for epoch in range(40):
    for (review, label) in training_data:
        # 1. Extract features
        phi = extractWordFeatures(review)
        
        # 2. Predict
        h = sigmoid(dotProduct(weights, phi))
        
        # 3. Update weights
        gradient = (h - label) * phi
        weights -= learning_rate * gradient
    
    # 4. Monitor overfitting
    train_error = evaluate(training_data)
    val_error = evaluate(validation_data)
```

**Key Observation**: Training error keeps falling with more epochs (7.4% at 20, 3.6% at 40, 1.5% at 70), but validation error stays around 26–28%. Extra epochs mostly memorize the training set.

### Milestone 5: Advanced Features

**Character N-Grams**:

Why character-level features matter for film criticism:

```python
# Word-level misses morphology
"Kafkaesque" → {'Kafkaesque': 1}  # Rare word, might not be in training

# Character 4-grams capture patterns
"Kafkaesque" → {'Kafk': 1, 'afka': 1, 'fkae': 1, 'kaes': 1, 'esque': 1}
# Now "-esque" pattern is learned across "Kafkaesque", "Chaplinesque", etc.
```

**Comparison**:
- `extractWordFeatures()`: Better for common words, interpretable weights
- `extractCharacterFeatures(4)`: Can pick up word parts like "-esque" across unseen words; on this dataset it did not beat word features (27.5% vs. 26.7% validation error, see `_eval/results.md`)

---

## 💡 Key Insights & Lessons Learned

### 1. Overfitting is Insidious

**Observation**: Validation error levels off around 26–27% after 40 epochs, while training error keeps decreasing. Character 5-grams go further: 0% training error, yet their validation F1 (0.733) is no better than word features (0.730).

**Lesson**: Always monitor validation curves—improving training accuracy ≠ better model.

**Solution**: Early stopping + regularization in future iterations.

### 2. Character N-Grams for Domain Jargon

**Why**: Film critics use specialized morphology:
- Suffixes: "-esque" (Kafkaesque, Chaplinesque)
- Prefixes: "pseudo-", "quasi-" (pseudo-intellectual, quasi-documentary)
- Word roots: "auteur", "noir", "mise-en-scène"

**Result**: On this dataset, character n-grams didn't beat word features overall (best validation F1: 0.733 for 5-grams vs. 0.730 for words), so the morphology benefit is still a hypothesis to test on jargon-heavy reviews.

### 3. Interpretability > Accuracy for Trust

**Stakeholder Feedback**: "We need to explain to angry fans why we classified Roger Ebert's review as negative."

**Lesson**: A 90% accurate black-box is useless if you can't audit decisions. Transparency builds trust.

**Implementation**: Export readable `weights` file showing word importance.

### 4. Domain Knowledge is Your Moat

**What AI Can't Do**:
- Understand that "competent but uninspired" is negative despite neutral words
- Know that "auteur theory" is film criticism jargon
- Recognize that "surprisingly good" often signals low expectations

**Your Advantage**: Combining technical skills with domain expertise creates systems that actually work in production.

---

## 🔄 Future Improvements

### Short-Term Enhancements

- [ ] **Confidence-based routing**: Auto-classify high-confidence predictions (>90%), flag low-confidence (50-70%) for human review
- [ ] **Multi-class extension**: Predict 1-5 star ratings using softmax regression
- [ ] **Active learning**: Retrain nightly on human-corrected misclassifications

### Long-Term Vision

- [ ] **Ensemble models**: Combine word-level + character n-gram predictions
- [ ] **Contextual embeddings**: Explore BERT/GPT for transfer learning (compare interpretability trade-offs)
- [ ] **Multi-language support**: Train language-specific models for international markets
- [ ] **A/B testing framework**: Measure real-world impact on Tomatometer update speed

---

## 🛠️ Development Setup

### Running Tests

```bash
# Run all test cases (should see 100/100)
python grader.py

# Expected output:
# ========== START GRADING ==========
# 3a-0: basic test ... PASS (10/10)
# 3b-0: basic sanity check ... PASS (10/10)
# ...
# ========== END GRADING [100/100 points] ==========
```

### Hyperparameter Tuning

```python
# In grader.py, modify test4a2():
weights = submission.learnPredictor(
    trainExamples, 
    validationExamples, 
    featureExtractor, 
    numEpochs=40,    # Try: 20, 40, 60, 80
    alpha=0.01       # Try: 0.001, 0.01, 0.1
)
```

**Observation** (error rates, alpha = 0.01):
- **epochs = 20**: train 7.4%, val 27.7%
- **epochs = 40** (default): train 3.6%, val 26.7%
- **epochs = 80**: train 1.2%, val 26.3%

Validation error barely moves after 40 epochs, while the gap to training error keeps widening.

### Debugging Misclassifications

```bash
# Generate detailed error analysis
python grader.py  # Outputs to error-analysis file

# Sample from error-analysis (a real miss):
# === nothing short of a masterpiece -- and a challenging one .
# Truth: 1, Prediction: -1 [WRONG]
# masterpiece    1 * 0.48 = 0.48
# challenging    1 * 0.28 = 0.28
# ...
# nothing        1 * -0.60 = -0.60
# short          1 * -0.83 = -0.83
# "nothing short of" is praise, but a bag-of-words model scores "nothing" and "short" as negative.
```

---

## 📚 Resources & References

### Dataset
- **Source**: [Rotten Tomatoes Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
- **Size**: 7,108 labeled reviews (3,554 train + 3,554 validation)
- **Labels**: Binary (+1 positive, -1 negative)

### Academic Background
- **Course**: [stanCode](https://stancode.tw/) SC201
- **Topics**: Supervised learning, optimization, NLP

### Key Papers
- **Logistic Regression**: [Statistical Learning Theory](https://web.stanford.edu/~hastie/ElemStatLearn/)
- **Sentiment Analysis**: [Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/omsa/omsa.pdf)

### Tools & Libraries
- **Python 3.7+**: Core language
- **Standard Library**: math, collections, random
- **No External Dependencies**: Pure Python implementation

---

## 🤝 Contributing

This is an educational project demonstrating ML fundamentals. While pull requests are welcome, the goal is to maintain implementation simplicity for learning purposes.

**Areas for contribution**:
- Documentation improvements
- Additional test cases
- Performance optimizations (maintaining interpretability)
- Visualization tools for weight analysis

---

## 📧 Contact

**Developer**: Chia-Chun Hung (Tony)  
**LinkedIn**: [Your LinkedIn]  
**Email**: [Your Email]  
**Portfolio**: [Your Website]

---

## 🙏 Acknowledgments

- **stanCode SC201** for the original assignment
- **Rotten Tomatoes** for the real-world use case
- **Film critics** whose reviews trained this model
- **Open-source community** for Python and educational resources

---

**Tech Stack**: Python • Machine Learning • NLP • Logistic Regression • Gradient Descent  
**Time Investment**: 15 hours  

---

*"Specialists are common. Translators are rare."*

**This project demonstrates both technical depth (from-scratch ML implementation) and domain expertise (film criticism understanding)—the combination that creates production-ready AI systems.**
