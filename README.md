# World Cup 2022 Prediction

An end-to-end machine learning pipeline for predicting international football match outcomes and estimating 2022 FIFA World Cup progression probabilities using historical match data, ELO ratings, recent form, probabilistic classification, and Monte Carlo simulation.

## Project Highlights

- **15,959 international matches** used for model development
- Historical data spanning **1950–2022**
- Leakage-safe pre-match feature engineering
- **3-class prediction:** Home Win / Draw / Away Win
- ELO ratings, recent form, goal trends, recency and competition weighting
- Chronological **walk-forward validation**
- Comparison of Logistic Regression, Random Forest and XGBoost
- Evaluation using **Accuracy, Log Loss and Brier Score**
- Final **90:10 Logistic Regression–XGBoost probability ensemble**
- **10,000 Monte Carlo World Cup simulations**
- Estimates of each team's probability of reaching the R16, QF, SF, Final and winning the tournament

---

## Problem Statement

Football match prediction is difficult because outcomes are uncertain and depend on many interacting factors.

This project builds a realistic pre-match prediction system that learns from historical international matches, constructs features using only information available before each match, predicts Home/Draw/Away probabilities, evaluates models chronologically, and converts match probabilities into tournament-level probabilities through simulation.

## Dataset

The project uses **15,959 international matches** from **26 February 1950 to 27 September 2022**. The final training cutoff is before the 2022 FIFA World Cup, so World Cup 2022 results are not used to train the prediction model.

Included data:

- `international_matches.csv` — historical international matches
- `2022_world_cup_groups.csv` — 2022 World Cup group information
- `2022_world_cup_matches.csv` — 2022 World Cup match information

## Methodology

```text
Historical Match Data
        ↓
Data Cleaning & Standardization
        ↓
Leakage-Safe Feature Engineering
        ↓
ELO + Form + Goal Features
        ↓
Home / Draw / Away Classification
        ↓
Walk-Forward Validation
        ↓
Model Comparison
        ↓
90:10 Logistic Regression + XGBoost Ensemble
        ↓
Match Probability Estimates
        ↓
10,000 Monte Carlo Tournament Simulations
        ↓
R16 → QF → SF → Final → Champion Probabilities
```

## Feature Engineering

The model uses pre-match information including:

- Home-team ELO
- Away-team ELO
- ELO difference
- Recent results and win rate
- Recent goals scored and conceded
- Recent goal difference
- Competition/tournament weighting
- Exponential recency weighting
- Home/away information

Features are constructed using information available before each match to reduce temporal leakage.

## Avoiding Data Leakage

A major focus of the project is avoiding temporal leakage.

For each historical match, features are generated using information available **before that match occurred**.

Validation is chronological:

```text
Past matches → Training
Future matches → Validation
```

This better reflects how the model would behave in a real forecasting setting than randomly shuffling historical matches.

## Models

Three models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

The models were evaluated using chronological walk-forward validation.

## Model Evaluation

Because the system outputs probabilities, accuracy alone is not sufficient.

- **Accuracy:** fraction of matches where the most probable class is correct.
- **Log Loss:** evaluates probability quality and strongly penalizes confident incorrect predictions.
- **Brier Score:** measures the squared error between predicted probabilities and the actual outcome.

### Walk-Forward Results

| Model | Accuracy | Log Loss | Brier Score |
|---|---:|---:|---:|
| **Logistic Regression** | **61.07%** | **0.8611** | **0.1681** |
| XGBoost | 60.36% | 0.8728 | 0.1704 |
| Random Forest | 58.64% | 0.8910 | 0.1753 |

Logistic Regression produced the strongest individual out-of-time performance across the evaluated metrics.

A **90:10 Logistic Regression–XGBoost probability ensemble** was selected for the final tournament prediction system.

## World Cup Simulation

The final model produces probability distributions rather than a deterministic tournament bracket.

The 2022 World Cup is simulated **10,000 times**. For each simulation:

1. Group-stage matches are simulated.
2. Group standings determine advancing teams.
3. Knockout matches are simulated.
4. The process continues through the Round of 16, Quarterfinals, Semifinals and Final.
5. Results are aggregated across simulations.

This estimates the probability of each team reaching each stage.

## 2022 World Cup Predictions

| Team | R16 | QF | SF | Final | Champion |
|---|---:|---:|---:|---:|---:|
| Brazil | 90.28% | 62.64% | 47.43% | 33.13% | **23.88%** |
| Argentina | 94.53% | 66.47% | 50.06% | 35.05% | **18.84%** |
| Spain | 75.88% | 46.08% | 22.00% | 11.49% | **6.60%** |
| Portugal | 75.33% | 39.14% | 23.22% | 11.83% | **6.56%** |
| Belgium | 74.76% | 41.87% | 19.25% | 9.92% | **5.46%** |
| France | 72.57% | 40.43% | 25.17% | 13.22% | **5.38%** |

Brazil was the highest-probability champion at **23.88%**, followed by Argentina at **18.84%**.

Argentina subsequently won the 2022 FIFA World Cup. This illustrates why probabilistic prediction is more appropriate than treating the highest-probability outcome as a certainty.

## Project Structure

```text
World-Cup-2022-Prediction/
│
├── refined_project.py
├── World_Cup_2022_Predictor_Refined.ipynb
├── World_Cup_2022_Predictor_Final.ipynb
│
├── international_matches.csv
├── 2022_world_cup_groups.csv
├── 2022_world_cup_matches.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

The `results/` directory is intentionally excluded from the clean repository package. Running the project generates the result files and visualizations.

## Installation

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd World-Cup-2022-Prediction
pip install -r requirements.txt
```

## Running the Project

Run the complete pipeline:

```bash
python refined_project.py
```

The script performs data loading, preprocessing, feature engineering, walk-forward validation, model comparison, final training, World Cup simulation and result generation.

Generated outputs are saved to:

```text
results/
```

## Running in Google Colab

Upload the repository ZIP or clone the GitHub repository in Colab, then run:

```python
!pip install -r requirements.txt
!python refined_project.py
```

Alternatively, open `World_Cup_2022_Predictor_Refined.ipynb` and run the notebook from top to bottom.

## Limitations

The model does not capture every factor influencing football matches, including:

- Player injuries and suspensions
- Starting lineups
- Tactical changes
- Managerial decisions
- Player-level quality
- Weather
- In-game events
- Red cards and unexpected injuries
- Penalty shootout uncertainty

Future versions could incorporate FIFA rankings, player-level ratings, squad strength, player availability, expected goals (xG), richer venue effects, calibration studies, feature ablation and lineup information.

## Key Takeaways

This project demonstrates an end-to-end approach to a real-world probabilistic prediction problem:

- Historical sports data can be transformed into meaningful team-strength features.
- Temporal validation is important when predicting future events from historical data.
- Accuracy alone is insufficient when a model outputs probabilities.
- Ensemble models can combine complementary predictive behavior.
- Monte Carlo simulation can translate individual match probabilities into tournament-level uncertainty.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SciPy
- Matplotlib
- Monte Carlo Simulation

## Author

**Ranik Biswas**  
IIT Kanpur

## License

This project is intended for educational and portfolio purposes.
