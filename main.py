import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
import time
from mss import mss
import re
import random
import os
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_secondary_screen():
    """
    Get the secondary screen using mss
    """
    with mss() as sct:
        # List all monitors
        for i, monitor in enumerate(sct.monitors[1:], 1):  # Skip primary monitor
            logging.info(f"Found monitor {i}: {monitor}")
        
        # Select the first secondary monitor (monitor 2)
        if len(sct.monitors) < 2:
            logging.error("No secondary monitor found!")
            return None
        
        return sct.monitors[2]  # mss uses 1-based indexing, 2 is second monitor

def extract_game_text(frame):
    """
    Extract both ongoing multipliers and red 'FLEW AWAY!' multiplier
    """
    height, width = frame.shape[:2]
    
    # Define a more precise center region
    center_y = height // 2
    center_x = width // 2
    
    crop_height = int(height * 0.3)  # 30% of height
    crop_width = int(width * 0.2)   # 20% of width
    
    # Calculate crop coordinates with offset to the right
    x_offset = int(width * 0.1)  # Shift right by 10% of screen width
    y_start = center_y - crop_height//2
    y_end = center_y + crop_height//2
    x_start = center_x - crop_width//2 + x_offset  # Add offset here
    x_end = center_x + crop_width//2 + x_offset    # And here
    
    # Ensure we don't go beyond screen boundaries
    x_start = min(max(0, x_start), width - crop_width)
    x_end = min(x_start + crop_width, width)
    
    # Crop the center region more precisely
    center_region = frame[y_start:y_end, x_start:x_end]
    
    # Save all processing steps for debugging
    timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
    # cv2.imwrite(f'debug/crop_{timestamp}.png', center_region)
    
    # First check for "FLEW AWAY!" text in white
    gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(f'debug/white_thresh_{timestamp}.png', white_thresh)
    
    # Increase image size for better OCR accuracy
    white_thresh_large = cv2.resize(white_thresh, (white_thresh.shape[1]*3, white_thresh.shape[0]*3))
    # cv2.imwrite(f'debug/white_thresh_large_{timestamp}.png', white_thresh_large)
    
    # Look for "FLEW AWAY!" text with multiple configurations
    flew_configs = [
        r'--oem 3 --psm 6',  # Assume uniform block of text
        r'--oem 3 --psm 7',  # Treat the text as a single line
        r'--oem 3 --psm 8'   # Treat the text as a word
    ]
    
    flew_away_present = False
    for config in flew_configs:
        flew_text = pytesseract.image_to_string(white_thresh_large, config=config).strip().upper()
        if "FLEW" in flew_text and "AWAY" in flew_text:
            flew_away_present = True
            logging.info(f"Detected FLEW AWAY with config {config}: '{flew_text}'")
            break
    
    if flew_away_present:
        # If "FLEW AWAY!" is present, look for red multiplier text
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        # Red color range in HSV (adjusted for better detection)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create mask for red text
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        # cv2.imwrite(f'debug/red_mask_{timestamp}.png', red_mask)
        
        # Apply mask to get only red text
        red_text = cv2.bitwise_and(center_region, center_region, mask=red_mask)
        # cv2.imwrite(f'debug/red_text_{timestamp}.png', red_text)
        
        # Increase size of red text image for better OCR
        red_text_large = cv2.resize(red_text, (red_text.shape[1]*3, red_text.shape[0]*3))
        # cv2.imwrite(f'debug/red_text_large_{timestamp}.png', red_text_large)
        
        # Look for multiplier in red text with multiple configurations
        multiplier_configs = [
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.x',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.x',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789.x'
        ]
        
        for config in multiplier_configs:
            red_text_str = pytesseract.image_to_string(red_text_large, config=config).strip()
            red_multiplier_match = re.search(r'\d+\.?\d*x', red_text_str)
            if red_multiplier_match:
                multiplier = red_multiplier_match.group()
                logging.info(f"🔥 FLEW AWAY! Final Multiplier: {multiplier}")
                # Save both cropped and full frame for verification
                # timestamp = time.strftime("%Y%m%d-%H%M%S-%f")
                # cv2.imwrite(f'flew_away_full_{multiplier}_{timestamp}.png', frame)
                # cv2.imwrite(f'flew_away_crop_{multiplier}_{timestamp}.png', center_region)
                return "flew_away", multiplier, center_region

    # If no FLEW AWAY or no red multiplier, look for regular white multiplier
    text = pytesseract.image_to_string(white_thresh_large, config=r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.x').strip()
    multiplier_match = re.search(r'\d+\.?\d*x', text)
    
    if multiplier_match:
        multiplier = multiplier_match.group()
        return "multiplier", multiplier, center_region
    
    return None, None, None

def check_flew_away(frame):
    """
    Specifically check for 'FLEW AWAY!' text
    """
    height, width = frame.shape[:2]
    
    # Crop the upper part of the region where "FLEW AWAY!" appears
    upper_region = frame[0:height//2, :]
    
    # Convert to grayscale
    gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get white text
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Increase image size for better OCR
    thresh = cv2.resize(thresh, (thresh.shape[1]*2, thresh.shape[0]*2))
    
    # Check for "FLEW AWAY!" with different PSM modes
    flew_configs = [
        r'--oem 3 --psm 6',  # Assume uniform block of text
        r'--oem 3 --psm 7',  # Treat the text as a single line
        r'--oem 3 --psm 8',  # Treat the text as a word
    ]
    
    for config in flew_configs:
        text = pytesseract.image_to_string(thresh, config=config).strip().upper()
        if "FLEW" in text or "AWAY" in text:
            logging.info(f"Detected text: {text}")
            return True
            
    return False

def init_game_data():
    """
    Initialize or load existing game data CSV
    """
    try:
        df = pd.read_csv('game_data.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            'game_id',
            'start_time',
            'end_time',
            'duration',
            'final_multiplier',
            'multiplier_sequence',
            'max_multiplier',
            'crash_speed'  # Time between last multiplier and crash
        ])
    return df

def save_game_data(df):
    """
    Save game data to CSV
    """
    df.to_csv('game_data.csv', index=False)

def main():
    """
    Main function to capture and analyze secondary screen
    """
    # Create debug directory if it doesn't exist
    os.makedirs('debug', exist_ok=True)
    
    # Initialize game data
    game_data = init_game_data()
    current_game = {
        'start_time': None,
        'multipliers': [],
        'last_multiplier_time': None
    }
    game_id = len(game_data) + 1

    with mss() as sct:
        monitor = get_secondary_screen()
        if not monitor:
            return
        
        logging.info(f"Starting capture on secondary screen")
        logging.info("Press 'q' to quit")
        
        last_text_time = 0
        last_multiplier = None
        
        while True:
            # Capture the screen
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            current_time = time.time()
            
            # Process every 0.002 seconds (2ms) for faster multiplier updates
            if current_time - last_text_time >= 0.002:
                event_type, value, center_region = extract_game_text(frame)
                
                if event_type == "multiplier":
                    if value != last_multiplier:
                        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        logging.info(f"Current multiplier: {value}")
                        
                        # Start new game if this is first multiplier
                        if not current_game['start_time']:
                            current_game['start_time'] = current_time_str
                        
                        # Record multiplier with timestamp
                        current_game['multipliers'].append({
                            'time': current_time_str,
                            'value': float(value.replace('x', ''))
                        })
                        current_game['last_multiplier_time'] = current_time_str
                        
                        last_multiplier = value
                
                elif event_type == "flew_away":
                    # Game ended, record data
                    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # Get the final multiplier from the "flew away" value
                    final_multiplier = float(value.replace('x', ''))
                    
                    if current_game['multipliers']:  # If we have previous multipliers
                        multiplier_values = [m['value'] for m in current_game['multipliers']]
                        # Add the final "flew away" value if it's different from the last multiplier
                        if final_multiplier != multiplier_values[-1]:
                            current_game['multipliers'].append({
                                'time': end_time,
                                'value': final_multiplier
                            })
                            multiplier_values.append(final_multiplier)
                        
                        start_time = current_game['start_time']
                        last_time = current_game['last_multiplier_time']
                        
                        # Calculate crash speed (time between last multiplier and crash)
                        last_time_obj = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S.%f")
                        end_time_obj = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")
                        crash_speed = (end_time_obj - last_time_obj).total_seconds()
                        
                        # Add new game data
                        new_game = {
                            'game_id': game_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': (end_time_obj - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")).total_seconds(),
                            'final_multiplier': final_multiplier,  # Use the flew away value
                            'multiplier_sequence': ','.join(map(str, multiplier_values)),
                            'max_multiplier': max(multiplier_values),
                            'crash_speed': crash_speed
                        }
                        
                        game_data = pd.concat([game_data, pd.DataFrame([new_game])], ignore_index=True)
                        save_game_data(game_data)
                        game_id += 1
                        
                        logging.info(f"Game {game_id-1} recorded: Final multiplier {final_multiplier}x, Duration: {new_game['duration']:.2f}s")
                    
                    # Reset for next game
                    current_game = {
                        'start_time': None,
                        'multipliers': [],
                        'last_multiplier_time': None
                    }
                    last_multiplier = None
                    last_text_time = current_time + 1.0
                    continue
                
                last_text_time = current_time
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    logging.info("Shutting down...")

if __name__ == "__main__":
    main()