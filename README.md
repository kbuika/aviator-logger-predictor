# Aviator Game Predictor

## Overview
This project implements a machine learning-based prediction system for the Aviator game, a multiplier-based betting game where players need to cash out before the "plane" flies away. The project consists of two main components:
- `main.py`: Real-time game state monitoring and data collection
- `predictor.py`: ML-based prediction and automated betting strategy

## Game Mechanics
The Aviator game operates on the following principles:
1. Each round starts with a multiplier of 1.00x
2. The multiplier increases over time
3. Players must place bets before the round starts
4. Players must cash out before the plane "flies away" to win
5. Maximum of two active bets per round
6. If you don't cash out before the crash, you lose your bet

## Project Components

### Data Collection (`main.py`)
- Uses computer vision (OpenCV) to monitor game state
- Captures multiplier values in real-time
- Detects game end ("FLEW AWAY") events
- Records game data to CSV for analysis
- Implements robust error handling and logging

### Prediction System (`predictor.py`)
- Uses XGBoost for crash point prediction
- Implements dynamic betting strategies
- Features adaptive risk management
- Tracks performance and maintains session history
- Includes self-learning capabilities

## Implemented Strategies

1. **Conservative Base Strategy**
   - Lower base target multiplier (1.3x-1.5x)
   - Small bet sizes (2.5-3.5% of balance)
   - Reset on losses

2. **Progressive Target Strategy**
   - Increases target multiplier after wins
   - Resets to base target after losses
   - Maintains win streak tracking

3. **Adaptive Risk Management**
   - Adjusts bet sizes based on performance
   - Scales risk with win streaks
   - Implements maximum risk caps

4. **ML-Based Prediction**
   - Uses recent game history
   - Considers time-series features
   - Adapts to changing patterns

## Challenges and Limitations

1. **Game Unpredictability**
   - Crash points are server-determined
   - No guaranteed patterns
   - High variance in outcomes

2. **Technical Challenges**
   - OCR reliability
   - Network latency issues
   - Timing of bet placement

3. **Strategy Limitations**
   - Risk of consecutive losses
   - Balance between risk and reward
   - Psychological factors

## Potential Future Strategies

1. **Pattern-Based Approaches**
   - Time-of-day analysis
   - Session pattern recognition
   - Streak analysis

2. **Multi-Model Ensemble**
   - Combine multiple prediction models
   - Weight predictions by confidence
   - Adaptive model selection

3. **Advanced Risk Management**
   - Kelly Criterion variations
   - Dynamic portfolio management
   - Recovery strategies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kbuika/aviator-logger.git
cd aviator-logger
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- For MacOS: `brew install tesseract`
- For Ubuntu: `sudo apt-get install tesseract-ocr`
- For Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

4. Configure your display settings:
- Ensure the game window is visible on a secondary monitor
- Adjust screen resolution if needed

## Usage

1. Start data collection:
```bash
python main.py
```

2. In a separate terminal, start the predictor:
```bash
python predictor.py
```

## Results and Observations

1. **Win Rate Patterns**
   - Higher success with conservative targets (1.3x-1.5x)
   - Progressive strategies show promise but higher risk
   - Win streaks often followed by sudden crashes

2. **Risk Management Effectiveness**
   - Small bet sizes (2-3%) show better longevity
   - Progressive betting requires strict discipline
   - Recovery strategies often lead to larger losses

3. **Prediction Accuracy**
   - Short-term patterns exist but unreliable
   - ML models show slight edge over random
   - Time-series features most influential

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Disclaimer
This project is for educational purposes only. Gambling can be addictive and risky. Never bet money you cannot afford to lose.

## License
MIT License - see LICENSE file for details 