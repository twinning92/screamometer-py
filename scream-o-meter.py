import time
import math
import numpy as np
import pyaudio
import board
import neopixel
import soundfile as sf
import threading
import digitalio


# Constants for NeoPixel
LED_PIN = board.D12
NUM_LEDS = 144
BRIGHTNESS = 0.1  # Scale 0-1
led_strip = neopixel.NeoPixel(LED_PIN, NUM_LEDS, brightness=BRIGHTNESS, auto_write=False)

# PyAudio constants
SAMPLE_RATE = 44100
CHANNELS = 2
BUFFER_SIZE = 1024  # Size of buffer for PyAudio

# Initialize variables for tracking high score
high_score = 0
session_high_score = 0
last_high_score_time = 0
inactivity_period = 10  # Seconds of inactivity to reset session high score and audio reset
noise_threshold = 2000.0  # Noise floor to filter out background noise
scaling_factor = 30.0  # Logarithmic scaling factor
alpha = 0.02  # Smoothing factor


# Initialize PyAudio for capturing microphone input
p = pyaudio.PyAudio()  # Initialize a single PyAudio instance for both input and output

# Function to apply logarithmic scaling
def apply_logarithmic_scaling(input_value, max_value, scaling_factor):
    input_value = max(input_value, 1.0)  # Avoid log(0)
    log_value = math.log(input_value) / math.log(scaling_factor)
    return int(min(max_value, log_value * (max_value / math.log(scaling_factor))))

def set_pixel_brightness(pixel_index, colour, brightness):
    brightness = max(0.0, min(brightness, 1.0))
    scaled_colour = tuple(int(c * brightness) for c in colour)
    led_strip[pixel_index] = scaled_colour

# Function to update the LED strip
def update_leds(display_value, high_score, session_high_score):
    led_strip.fill((0, 0, 0))  # Clear the LED strip

    # Set LEDs up to the display value to blue
    for i in range(display_value):
        set_pixel_brightness(i, (127, 51, 0), 0.8)  

    # Set session high score LED to green
    if session_high_score < NUM_LEDS:
        set_pixel_brightness(session_high_score, (0, 255, 0), 1.0)  # Green

    # Set high score LED to red
    if high_score < NUM_LEDS:
        set_pixel_brightness(high_score, (255, 0, 0), 1.0)  # Red
        set_pixel_brightness(high_score + 1, (255, 0, 0), 1.0)  # Red
    led_strip.show()  

def motion_sense():
    pir = digitalio.DigitalInOut(board.D13)
    pir.direction = digitalio.Direction.INPUT
    
    while True:
        if pir.value:
            play_audio()
            print("Motion Detected")
            time.sleep(5)

def play_audio():
    file = "../test_recording.wav"
    audio_data, sample_rate = sf.read(file, dtype='float32')
    current_frame = 0

    def callback(out_data, frame_count, time_info, status):
        nonlocal current_frame
        end = current_frame + frame_count

        # Check if we're at the end of the audio file
        if end > len(audio_data):
            data_chunk = audio_data[current_frame:]  # Get remaining data
            return (data_chunk.tobytes(), pyaudio.paComplete)  # Signal end of stream

        # If we have enough data, return the chunk
        data_chunk = audio_data[current_frame:end]
        current_frame = end
        return (data_chunk.tobytes(), pyaudio.paContinue)

    stream_out = p.open(format=pyaudio.paFloat32,
                        channels=audio_data.shape[1] if len(audio_data.shape) > 1 else 1,
                        rate=sample_rate,
                        output=True,
                        stream_callback=callback)
    
    stream_out.start_stream()

    while stream_out.is_active():
        time.sleep(0.1)  # Avoid high CPU usage

    stream_out.stop_stream()
    stream_out.close()

# Function to capture microphone input and process data
def scream_o_meter():
    global high_score, session_high_score, last_high_score_time

    stream_in = p.open(format=pyaudio.paInt16,
                       channels=CHANNELS,
                       rate=SAMPLE_RATE,
                       input=True,
                       input_device_index=1,
                       frames_per_buffer=BUFFER_SIZE)

    smoothed_value = 0

    while True:
        data = stream_in.read(BUFFER_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Calculate mean of absolute values (signal amplitude)
        mean = np.mean(np.abs(audio_data))

        # Apply noise threshold
        if mean < noise_threshold:
            mean = 0
        
        # Apply logarithmic scaling
        mapped_value = apply_logarithmic_scaling(mean, NUM_LEDS, scaling_factor)

        # Exponential smoothing
        smoothed_value = (alpha * mapped_value) + ((1 - alpha) * smoothed_value)
        display_value = int(smoothed_value)

        # Update high scores
        if display_value > high_score:
            high_score = display_value

        if display_value > session_high_score:
            session_high_score = display_value
            last_high_score_time = time.time()

        # Reset session high score after inactivity
        if time.time() - last_high_score_time > inactivity_period:
            session_high_score = 0

        # Update LEDs
        update_leds(display_value, high_score, session_high_score)

        time.sleep(0.01)  # Small delay to avoid CPU overload

# Run the scream-o-meter and audio playback in separate threads
if __name__ == "__main__":
    try:
        # Run play_audio and scream_o_meter in parallel using threads
        motion_thread = threading.Thread(target=motion_sense)
        scream_thread = threading.Thread(target=scream_o_meter)

        motion_thread.start()
        scream_thread.start()

        # Wait for both threads to complete
        motion_thread.join()
        scream_thread.join()

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        led_strip.fill((0, 0, 0))  # Clear LEDs
        led_strip.show()
        p.terminate()  # Terminate PyAudio
