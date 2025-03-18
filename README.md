# Aviator Game Predictor
NOTE: I "vibe-coded" through this project;
Tldr: Aviator is extremely random and is a zero-sum game except for the fact that you are the one who will keep losing! Please don't even try it.

### Caveats
- The `main.py` script is designed such that Aviator runs on a secondary monitor with the comments section closed. You may need to adjust this.
- You can uncomment the `cv2.imwrite()` lines to log screenshots for debugging. 

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

### Core Functionality
- Real-time game monitoring and crash point prediction
- Dynamic bet sizing based on confidence levels
- Adaptive strategy adjustment based on performance
- Comprehensive logging and performance tracking
- Automated risk management

### Data Collection (`main.py`)
- Uses computer vision (OpenCV) to monitor game state
- Captures multiplier values in real-time
- Detects game end ("FLEW AWAY") events
- Records game data to CSV for analysis
- Implements robust error handling and logging

### Risk Management
- Kelly Criterion implementation
- Dynamic risk adjustment
- Balance protection mechanisms
- Progressive bet sizing
- Win streak factoring

### Performance Tracking
- Detailed game logging in CSV format
- Real-time performance metrics
- Skip analysis and evaluation
- ROI tracking
- Win rate analysis

## Requirements

```
pandas
numpy
xgboost
scikit-learn
matplotlib
seaborn
opencv-python
pytesseract
mss
```

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
This will start monitoring games and collecting crash data.

2. Run the predictor:
```bash
python predictor.py
```
This will start the automated betting system.

## Configuration

### Betting Parameters
- Initial Balance: $1000 (configurable)
- Base Target: 1.5x (adaptive)
- Maximum Target: 3.0x
- Base Risk: 3.5% per trade
- Maximum Risk: 5% per trade

### Confidence Levels
- High (80-100%): Up to 50% larger bets
- Medium (65-80%): Up to 30% larger bets
- Low (50-65%): 50-100% of base bet size
- Below 50%: Skip game

### Strategy Adaptation
- Analyzes every 5 games
- Adjusts targets based on win rate
- Modifies risk based on balance trend
- Updates confidence calculation based on performance

## Files

- `main.py`: Game monitoring and data collection
- `predictor.py`: Betting strategy and prediction logic
- `game_data.csv`: Raw game data
- `games_played.csv`: Betting history and results
- `trading_performance.png`: Performance visualization

## Performance Metrics

The system tracks:
- Win/Loss ratio
- ROI per session
- Total profit/loss
- Skip accuracy
- Confidence correlation
- Risk exposure

## Safety Features

1. Balance Protection
   - Reduces risk when balance drops below 80% of initial
   - Increases risk when performing well
   - Maximum bet size limits

2. Risk Management
   - Kelly Criterion for optimal bet sizing
   - Dynamic risk adjustment
   - Win streak consideration
   - Volatility-based risk scaling

3. Loss Prevention
   - Skip low-confidence games
   - Progressive target adjustment
   - Automatic session reset on depletion

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading involves risk of monetary loss. Past performance is not indicative of future results. 
