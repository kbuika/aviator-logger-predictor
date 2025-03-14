import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class AviatorPredictor:
    def __init__(self, initial_balance=1000):
        self.model = XGBRegressor(
            n_estimators=150,  # Increased for better learning
            learning_rate=0.15,  # Slightly more aggressive learning
            max_depth=6,  # Deeper trees for more complex patterns
            random_state=42,
            early_stopping_rounds=10
        )
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.trades = []
        self.games_played = self.load_games_played()
        self.session_number = len(self.games_played) + 1
        self.active_bets = []
        self.waiting_for_next_game = True
        self.last_crash_points = []
        
        # More aggressive base parameters
        self.base_conservative_target = 1.5  # Increased base target
        self.current_conservative_target = self.base_conservative_target
        self.target_increment = 0.2  # Faster target increase
        self.win_streak = 0
        self.max_target = 3.0  # Maximum target multiplier
        
        # Adjusted risk parameters
        self.conservative_risk = 0.035  # Increased base risk
        self.max_risk = 0.05  # Maximum risk per trade
        self.crash_mean = None
        self.crash_std = None
        self.crash_median = None
        
        # Load historical data and train model
        self.initialize_model()

    def load_games_played(self):
        """
        Load or create games_played.csv
        """
        try:
            return pd.read_csv('games_played.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=[
                'game_id',
                'timestamp',
                'crash_point',
                'bet1_amount',
                'bet1_target',
                'bet1_type',
                'bet1_result',
                'bet1_profit',
                'bet2_amount',
                'bet2_target',
                'bet2_type',
                'bet2_result',
                'bet2_profit',
                'total_profit',
                'balance_after',
                'session_number'
            ])
            df.to_csv('games_played.csv', index=False)
            return df
            
    def save_session(self, strategy_used="default"):
        """
        Save session results to games_played.csv
        """
        if not self.trades:
            return
            
        df_trades = pd.DataFrame(self.trades)
        session_data = {
            'session_number': self.session_number,
            'start_balance': self.initial_balance,
            'end_balance': self.balance,
            'trades_made': len(self.trades),
            'win_rate': df_trades['success'].mean(),
            'start_time': df_trades['timestamp'].min(),
            'end_time': df_trades['timestamp'].max(),
            'duration_minutes': (df_trades['timestamp'].max() - df_trades['timestamp'].min()).total_seconds() / 60,
            'strategy_used': strategy_used
        }
        
        self.games_played = pd.concat([self.games_played, pd.DataFrame([session_data])], ignore_index=True)
        self.games_played.to_csv('games_played.csv', index=False)
        
    def reset_session(self):
        """
        Reset the session with new balance and increment session number
        """
        self.save_session()
        self.balance = self.initial_balance
        self.trades = []
        self.session_number += 1
        logging.info(f"Starting new session {self.session_number} with balance ${self.initial_balance}")

    def prepare_features(self, df):
        """
        Prepare features for the model from game history
        """
        # Convert multiplier sequences to lists
        df['multiplier_list'] = df['multiplier_sequence'].apply(lambda x: [float(i) for i in x.split(',')])
        
        # Create features
        features = []
        labels = []
        
        for idx, row in df.iterrows():
            multipliers = row['multiplier_list']
            
            # Skip sequences that are too short
            if len(multipliers) < 5:
                continue
            
            # Create features from different points in the sequence
            for i in range(5, len(multipliers)):
                feature_set = {
                    'last_5_avg': np.mean(multipliers[i-5:i]),
                    'last_5_std': np.std(multipliers[i-5:i]),
                    'last_5_min': np.min(multipliers[i-5:i]),
                    'last_5_max': np.max(multipliers[i-5:i]),
                    'current_value': multipliers[i-1],
                    'growth_rate': multipliers[i-1] / multipliers[i-2] if i > 1 else 1,
                    'time_of_day': pd.to_datetime(row['start_time']).hour,
                    'duration_so_far': i * 0.1,  # Assuming 0.1s between multipliers
                }
                features.append(feature_set)
                labels.append(row['final_multiplier'])
        
        return pd.DataFrame(features), np.array(labels)

    def train(self, csv_path='game_data.csv'):
        """
        Train the model on historical data
        """
        # Load and prepare data
        df = pd.read_csv(csv_path)
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f"Model MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        return mse, mae

    def predict_crash_point(self, current_sequence):
        """
        Predict the crash point given the current sequence of multipliers
        """
        if len(current_sequence) < 5:
            return None
        
        # Prepare features for current sequence
        feature_set = {
            'last_5_avg': np.mean(current_sequence[-5:]),
            'last_5_std': np.std(current_sequence[-5:]),
            'last_5_min': np.min(current_sequence[-5:]),
            'last_5_max': np.max(current_sequence[-5:]),
            'current_value': current_sequence[-1],
            'growth_rate': current_sequence[-1] / current_sequence[-2],
            'time_of_day': datetime.now().hour,
            'duration_so_far': len(current_sequence) * 0.1,
        }
        
        X = pd.DataFrame([feature_set])
        predicted_crash = self.model.predict(X)[0]
        return predicted_crash

    def initialize_model(self):
        """
        Initialize model with both historical game data and played games
        """
        try:
            # Load both data sources
            game_data = pd.read_csv('game_data.csv')
            games_played = pd.read_csv('games_played.csv')
            
            # Analyze crash points from both sources
            if not games_played.empty:
                recent_crashes = games_played['crash_point'].tail(50)
                self.crash_mean = recent_crashes.mean()
                self.crash_std = recent_crashes.std()
                self.crash_median = recent_crashes.median()
                
                # Adjust base target based on historical performance
                winning_targets = games_played[games_played['bet1_result'] == 'win']['bet1_target']
                if not winning_targets.empty:
                    self.base_conservative_target = max(1.5, winning_targets.mean() * 0.9)
                    self.current_conservative_target = self.base_conservative_target
            
            # Train model on game data
            if not game_data.empty:
                self.train('game_data.csv')
                
            logging.info(f"Model initialized with base target {self.base_conservative_target:.2f}x")
            
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")

    def calculate_bet_size(self, confidence, max_risk=0.05):
        """
        Calculate bet size with more aggressive scaling
        """
        # Scale risk based on win streak
        adjusted_risk = min(self.conservative_risk * (1 + self.win_streak * 0.1), self.max_risk)
        
        # Increase risk if we're significantly up
        if self.balance > self.initial_balance * 1.2:
            adjusted_risk *= 1.2
        
        # Calculate base bet size
        max_bet = self.balance * adjusted_risk
        kelly_bet = self.balance * confidence * adjusted_risk
        base_bet = min(kelly_bet, max_bet)
        
        # Scale bet size based on recent performance
        if len(self.trades) >= 5:
            recent_performance = np.mean([t['success'] for t in self.trades[-5:]])
            if recent_performance > 0.6:
                base_bet *= 1.3  # More aggressive scaling on wins
            elif recent_performance < 0.4:
                base_bet *= 0.7
        
        # Never bet more than max_risk of initial balance
        return min(base_bet, self.initial_balance * self.max_risk)

    def decide_exit_point(self, predicted_crash, current_value):
        """
        Decide when to exit based on prediction and risk management
        """
        if predicted_crash <= current_value:
            return current_value
        
        # Conservative exit at 80% of predicted crash
        safe_exit = current_value + (predicted_crash - current_value) * 0.8
        return safe_exit

    def simulate_trade(self, current_sequence, actual_crash):
        """
        Simulate a trade with the current model and strategy during an active game
        """
        if len(current_sequence) < 5:
            return
            
        # Calculate confidence based on recent prediction accuracy
        confidence = 0.5  # Start with base confidence
        if len(self.trades) > 0:
            recent_accuracy = np.mean([t['success'] for t in self.trades[-10:]])
            confidence = (confidence + recent_accuracy) / 2
        
        # Calculate bet size
        bet_size = self.calculate_bet_size(confidence)
        
        # Decide exit point (more conservative now)
        exit_point = current_sequence[-1] + 0.5  # Simple fixed increment for now
        
        # Simulate trade result - success only if we exit before crash
        success = exit_point < actual_crash  # Changed <= to < to ensure we exit before crash
        profit = bet_size * (exit_point - 1) if success else -bet_size
        
        # Update balance
        self.balance += profit
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'bet_size': bet_size,
            'exit_point': exit_point,
            'actual_crash': actual_crash,
            'profit': profit,
            'success': success,
            'balance': self.balance,
            'session_number': self.session_number,
            'bet_type': 'conservative'  # Default type for now
        }
        self.trades.append(trade)
        
        logging.info(f"Session {self.session_number} - Trade completed: Exit@{exit_point:.2f}x, Actual@{actual_crash:.2f}x, Profit: ${profit:.2f}, Balance: ${self.balance:.2f}")
        
        # Check if we need to reset
        if self.balance <= 0:
            logging.info("Balance depleted. Saving session and retraining model...")
            self.save_session()
            return "RESET"

    def plot_performance(self):
        """
        Plot trading performance focusing on balance over time
        """
        if not self.trades:
            logging.info("No trades to plot")
            return
        
        df_trades = pd.DataFrame(self.trades)
        
        plt.figure(figsize=(12, 6))
        
        # Balance over time
        plt.plot(df_trades.index, df_trades['balance'], 'b-', label='Balance')
        plt.title('Trading Performance')
        plt.xlabel('Trade Number')
        plt.ylabel('Balance ($)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('trading_performance.png')
        plt.close()

        # Log some basic statistics
        win_rate = df_trades['success'].mean()
        total_profit = df_trades['profit'].sum()
        avg_profit = df_trades['profit'].mean()
        
        logging.info(f"Performance Stats - Win Rate: {win_rate:.2%}, "
                    f"Total Profit: ${total_profit:.2f}, "
                    f"Average Profit: ${avg_profit:.2f}")

    def analyze_strategy_performance(self):
        """
        Analyze historical betting performance to optimize strategy
        """
        try:
            df = pd.read_csv('games_played.csv')
            if len(df) < 10:  # Need minimum games for analysis
                return
                
            # Analyze conservative strategy
            conservative_bets = df[df['bet1_type'] == 'conservative']
            conservative_stats = {
                'win_rate': conservative_bets['bet1_result'].value_counts().get('win', 0) / len(conservative_bets),
                'avg_profit': conservative_bets['bet1_profit'].mean(),
                'best_targets': conservative_bets[conservative_bets['bet1_result'] == 'win']['bet1_target'].mean()
            }
            
            # Analyze aggressive strategy
            aggressive_bets = df[df['bet2_type'] == 'aggressive']
            aggressive_stats = {
                'win_rate': aggressive_bets['bet2_result'].value_counts().get('win', 0) / len(aggressive_bets),
                'avg_profit': aggressive_bets['bet2_profit'].mean(),
                'best_targets': aggressive_bets[aggressive_bets['bet2_result'] == 'win']['bet2_target'].mean()
            }
            
            # Analyze crash points distribution
            recent_crashes = df['crash_point'].tail(50)  # Last 50 games
            crash_stats = {
                'mean': recent_crashes.mean(),
                'std': recent_crashes.std(),
                'median': recent_crashes.median(),
                'q25': recent_crashes.quantile(0.25),
                'q75': recent_crashes.quantile(0.75)
            }
            
            # Update strategy parameters based on analysis
            self.update_strategy_parameters(conservative_stats, aggressive_stats, crash_stats)
            
            logging.info(f"Strategy Analysis - Conservative WR: {conservative_stats['win_rate']:.2%}, "
                        f"Aggressive WR: {aggressive_stats['win_rate']:.2%}")
            
        except Exception as e:
            logging.error(f"Error analyzing strategy: {str(e)}")
    
    def update_strategy_parameters(self, conservative_stats, aggressive_stats, crash_stats):
        """
        Update betting parameters based on historical performance
        """
        # Adjust conservative strategy
        if conservative_stats['win_rate'] < 0.5:
            # If winning rate is low, be more conservative
            self.current_conservative_target = self.base_conservative_target * 0.9  # Reduce target multiplier
            self.conservative_risk = 0.02  # Reduce risk
        elif conservative_stats['win_rate'] > 0.7:
            # If winning rate is high, be slightly more aggressive
            self.current_conservative_target = self.base_conservative_target * 1.1
            self.conservative_risk = 0.03
        else:
            # Balanced approach
            self.current_conservative_target = self.base_conservative_target
            self.conservative_risk = 0.025
            
        # Adjust aggressive strategy
        if aggressive_stats['win_rate'] < 0.3:
            # If aggressive strategy is failing, be more conservative
            self.current_conservative_target = self.base_conservative_target * 0.8
            self.conservative_risk = 0.015
        elif aggressive_stats['win_rate'] > 0.5:
            # If aggressive strategy is working well
            self.current_conservative_target = self.base_conservative_target * 1.2
            self.conservative_risk = 0.025
        else:
            # Balanced approach
            self.current_conservative_target = self.base_conservative_target
            self.conservative_risk = 0.02
            
        # Update crash point prediction factors
        self.crash_mean = crash_stats['mean']
        self.crash_std = crash_stats['std']
        self.crash_median = crash_stats['median']
        
    def place_bets_for_next_game(self, last_crash_point):
        """
        Place bets with dynamic target adjustment and aggressive scaling
        """
        if not self.waiting_for_next_game:
            return []
            
        self.last_crash_points.append(last_crash_point)
        if len(self.last_crash_points) > 20:
            self.last_crash_points.pop(0)
        
        # More frequent strategy analysis
        if len(self.trades) % 5 == 0 and len(self.trades) > 0:  # Analyze every 5 games
            logging.info("Re-evaluating strategy...")
            self.analyze_strategy_performance()
            self.plot_performance()
        
        if len(self.last_crash_points) < 5:
            return []
        
        # Calculate confidence with aggressive bias
        confidence = 0.6  # Start with higher base confidence
        if len(self.trades) > 0:
            recent_accuracy = np.mean([t['success'] for t in self.trades[-10:]])
            confidence = (confidence + recent_accuracy) / 2
            
            # Boost confidence on win streaks
            if self.win_streak > 2:
                confidence *= 1.1
        
        bets = []
        available_balance = self.balance
        
        # Place bet with dynamic target
        if available_balance > 0:
            bet_size = self.calculate_bet_size(confidence)
            if bet_size <= available_balance:
                # Cap the target multiplier
                target = min(self.current_conservative_target, self.max_target)
                bets.append({
                    'amount': bet_size,
                    'target_multiplier': target,
                    'type': 'conservative'
                })
        
        if bets:
            self.active_bets = bets
            self.waiting_for_next_game = False
            total_risk = sum(bet['amount'] for bet in bets)
            logging.info(f"Placed bet with target {self.current_conservative_target:.2f}x "
                        f"(Win streak: {self.win_streak}, Risk: {(total_risk/self.balance)*100:.1f}%)")
        
        return bets

    def process_game_result(self, crash_point):
        """
        Process the game result and update dynamic target multiplier
        """
        if not self.active_bets:
            return
            
        total_profit = 0
        bet_results = []
        
        for bet in self.active_bets:
            success = crash_point > bet['target_multiplier']
            profit = bet['amount'] * (bet['target_multiplier'] - 1) if success else -bet['amount']
            total_profit += profit
            
            # Update dynamic target multiplier based on result
            if success:
                self.win_streak += 1
                self.current_conservative_target += self.target_increment
                logging.info(f"Win! Increasing target to {self.current_conservative_target:.2f}x")
            else:
                self.win_streak = 0
                self.current_conservative_target = self.base_conservative_target
                logging.info(f"Loss! Resetting target to {self.base_conservative_target:.2f}x")
            
            bet_results.append({
                'amount': bet['amount'],
                'target': bet['target_multiplier'],
                'type': bet['type'],
                'result': 'win' if success else 'loss',
                'profit': profit
            })
            
            # Record trade for performance tracking
            trade = {
                'timestamp': datetime.now(),
                'bet_size': bet['amount'],
                'exit_point': bet['target_multiplier'],
                'actual_crash': crash_point,
                'profit': profit,
                'success': success,
                'balance': self.balance + profit,
                'session_number': self.session_number,
                'bet_type': bet['type'],
                'win_streak': self.win_streak
            }
            self.trades.append(trade)
            self.balance += profit
            
            logging.info(f"Game ended: Crash@{crash_point:.2f}x, Target@{bet['target_multiplier']:.2f}x, "
                        f"Profit: ${profit:.2f}, Balance: ${self.balance:.2f}")
        
        # Record game in games_played.csv
        game_record = {
            'game_id': len(self.games_played) + 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'crash_point': crash_point,
            'bet1_amount': bet_results[0]['amount'] if len(bet_results) > 0 else 0,
            'bet1_target': bet_results[0]['target'] if len(bet_results) > 0 else 0,
            'bet1_type': bet_results[0]['type'] if len(bet_results) > 0 else 'none',
            'bet1_result': bet_results[0]['result'] if len(bet_results) > 0 else 'none',
            'bet1_profit': bet_results[0]['profit'] if len(bet_results) > 0 else 0,
            'bet2_amount': bet_results[1]['amount'] if len(bet_results) > 1 else 0,
            'bet2_target': bet_results[1]['target'] if len(bet_results) > 1 else 0,
            'bet2_type': bet_results[1]['type'] if len(bet_results) > 1 else 'none',
            'bet2_result': bet_results[1]['result'] if len(bet_results) > 1 else 'none',
            'bet2_profit': bet_results[1]['profit'] if len(bet_results) > 1 else 0,
            'total_profit': total_profit,
            'balance_after': self.balance,
            'session_number': self.session_number
        }
        
        self.games_played = pd.concat([self.games_played, pd.DataFrame([game_record])], ignore_index=True)
        self.games_played.to_csv('games_played.csv', index=False)
        
        # Reset for next game
        self.active_bets = []
        self.waiting_for_next_game = True
        
        # Check if we need to reset
        if self.balance <= 0:
            logging.info("Balance depleted. Saving session and retraining model...")
            self.save_session()
            return "RESET"

def main():
    # Create games_played.csv if it doesn't exist
    if not os.path.exists('games_played.csv'):
        initial_df = pd.DataFrame(columns=[
            'game_id',
            'timestamp',
            'crash_point',
            'bet1_amount',
            'bet1_target',
            'bet1_type',
            'bet1_result',
            'bet1_profit',
            'bet2_amount',
            'bet2_target',
            'bet2_type',
            'bet2_result',
            'bet2_profit',
            'total_profit',
            'balance_after',
            'session_number'
        ])
        initial_df.to_csv('games_played.csv', index=False)
        logging.info("Created new games_played.csv file with initial columns")

    predictor = AviatorPredictor(initial_balance=1000)
    
    try:
        while True:  # Outer loop for sessions
            if not os.path.exists('game_data.csv'):
                logging.error("game_data.csv not found! Please run the main script first to collect some data.")
                return
                
            df = pd.read_csv('game_data.csv')
            if len(df) == 0:
                logging.error("No data found in game_data.csv!")
                return
            
            logging.info(f"Starting session {predictor.session_number}...")
            last_game_id = None
            
            while True:
                try:
                    # Read latest game data
                    df = pd.read_csv('game_data.csv')
                    if len(df) == 0:
                        time.sleep(0.1)
                        continue
                    
                    latest_game = df.iloc[-1]
                    current_game_id = latest_game['game_id']
                    
                    # New game completed
                    if current_game_id != last_game_id and last_game_id is not None:
                        # Process previous game result
                        crash_point = float(latest_game['final_multiplier'])
                        result = predictor.process_game_result(crash_point)
                        
                        if result == "RESET":
                            predictor.reset_session()
                            break  # Break inner loop to restart session
                        
                        # Place bets for next game
                        predictor.place_bets_for_next_game(crash_point)
                    
                    last_game_id = current_game_id
                    
                    # Update performance plot every 10 trades
                    if len(predictor.trades) % 10 == 0 and predictor.trades:
                        predictor.plot_performance()
                    
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    predictor.save_session()
                    logging.info("Stopping prediction...")
                    return
                except Exception as e:
                    logging.error(f"Error during prediction: {str(e)}")
                    time.sleep(1)
                    continue
                    
    except Exception as e:
        logging.error(f"Initialization error: {str(e)}")
        predictor.save_session()

if __name__ == "__main__":
    main() 