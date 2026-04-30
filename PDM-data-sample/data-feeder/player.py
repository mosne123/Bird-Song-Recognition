# Import the pygame library for audio playback
import pygame
import serial
import os
import shutil
import time

# Initialize the pygame mixer
pygame.mixer.init()

def play_mp3(file_path):
    """Plays an MP3 file using pygame."""
    # attach to serial port at 115200 baud and send character 'p' to trigger the playback of the mp3 file on the microcontroller
    trigger_playback()
    
    try:
        # Load the MP3 file
        pygame.mixer.music.load(file_path)
        # Play the MP3 file
        pygame.mixer.music.play()
        time.sleep(30)
        print(f"Playing: {file_path}")

        # Wait for the music to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except pygame.error as e:
        print(f"Error playing file {file_path}: {e}")

def trigger_playback():
    """Sends the character 'p' to the microcontroller via serial port."""
    try:
        # Open the serial port at 115200 baud
        with serial.Serial('COM5', 115200, timeout=1) as ser:
            print("Connected to serial port.")
            # Send the character 'p'
            ser.write(b'p')
            print("Sent 'p' to trigger playback.")
    except serial.SerialException as e:
        print(f"Error: {e}")

def rename_last_played_file(directory, last_played_file):
    """Renames the file '00007.wav' in the /server/out directory to the name of the last played MP3 file."""
    try:
        # Correct the path to server/out relative to the Python script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_out_dir = os.path.join(script_dir, 'server', 'out')
        target_file = os.path.join(server_out_dir, '00007.wav')

        # Debugging: Print the paths being used
        print(f"Looking for '00007.wav' in: {server_out_dir}")

        if not os.path.exists(target_file):
            print("No file '00007.wav' found in /server/out.")
            return

        new_file_name = os.path.splitext(last_played_file)[0] + '.wav'  # Extracts the file name without extension
        new_file_path = os.path.join(server_out_dir, new_file_name)

        # Debugging: Print the renaming operation details
        print(f"Renaming '{target_file}' to '{new_file_path}'")

        shutil.move(target_file, new_file_path)
        print(f"Renamed '00007.wav' to '{new_file_name}' in /server/out.")

    except Exception as e:
        print(f"Error renaming file: {e}")

def play_all_mp3_in_directory(directory):
    """Plays all unique MP3 files in the specified directory."""
    try:
        # Get a list of all MP3 files in the directory
        mp3_files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
        
        if not mp3_files:
            print("No MP3 files found in the directory.")
            return

        for mp3_file in mp3_files:
            file_path = os.path.join(directory, mp3_file)
            print(f"Playing: {file_path}")
            play_mp3(file_path)

            # Add a 1-second delay before renaming the last played file
            time.sleep(1)
            rename_last_played_file(directory, mp3_file)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage
    directory = input("Enter the directory path to look for MP3 files: ")
    play_all_mp3_in_directory(directory)
