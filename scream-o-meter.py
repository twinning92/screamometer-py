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
BRIGHTNESS = 1  # Scale 0-1
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
scaling_factor = 5.0  # Logarithmic scaling factor
alpha = 0.04  # Smoothing factor


# Initialize PyAudio for capturing microphone input
p = pyaudio.PyAudio()  # Initialize a single PyAudio instance for both input and output

# Function to apply logarithmic scaling
def apply_logarithmic_scaling(input_value, max_value, scaling_factor):
    input_value = max(input_value, 1.0)  # Avoid log(0)
    log_value = math.log(input_value) / math.log(scaling_factor)
    return int(min(max_value, log_value * (max_value / math.log(scaling_factor))))

# Function to update the LED strip
def update_leds(display_value, high_score, session_high_score):
    # Clear the LED strip
    display_value = 143
    led_strip.fill((0, 0, 0))
    if display_value >= NUM_LEDS - 1:
        victory_animation()
        time.sleep(3)
        return
    # Determine the color based on the current display value
    if display_value >= NUM_LEDS * 0.95:
        led_colour = (0, 255, 0)  # white for very high values
    elif display_value >= NUM_LEDS * 0.8:
        led_colour = (255, 0, 0)      # red for high values
    elif display_value >= NUM_LEDS * 0.6:
        led_colour = (255, 0, 255)    # magenta for middle values
    else:
        led_colour = (255, 131, 0)    # orange for low values

    # Update LEDs according to the current amplitude value
    for i in range(min(display_value, NUM_LEDS)):
        led_strip[i] = led_colour

    # Ensure high score does not exceed the LED count minus space for indication
    clamped_high_score = min(high_score, NUM_LEDS - 3)
    clamped_session_high_score = min(session_high_score, NUM_LEDS - 1)

    # Highlight the session high score if within bounds
    if clamped_session_high_score < NUM_LEDS:
        led_strip[clamped_session_high_score] = (0, 255, 0)  # Green for session high score

    # Highlight the overall high score with a more prominent marker
    if clamped_high_score < NUM_LEDS - 2:  # Ensure there's room for the red markers
        led_strip[clamped_high_score] =     (255, 0, 0)  # Red for high score
        led_strip[clamped_high_score + 1] = (255, 0, 0)  # Continue the red marker
        led_strip[clamped_high_score + 2] = (255, 0, 0)  # Ensure it's distinctly visible
        

    
    led_strip.show()
    
    
def victory_animation():
    # blink_display()
    sweep_with_reverse_and_fade()
    
def blink_display():
    led_strip.fill((255,0,0))
    print("victory!")
    for _ in range(3):
        led_strip.fill((255,0,0))  # Turn all LEDs to the specified color
        led_strip.show()
        time.sleep(0.2)
        led_strip.fill((0, 0, 0))  # Turn off all LEDs
        led_strip.show()
        time.sleep(0.2)
        
def sweep_with_reverse_and_fade(color=(255, 0, 0), num_sweeps=3, target_fps=300):
    steps = 35  # Number of steps for the fade-out effect
    frame_duration = 1.0 / target_fps  # Duration of each frame in seconds

    for _ in range(num_sweeps):
        fading_leds = {}  # Dictionary to track fading LEDs: {LED index: remaining fade steps}
        total_frames = NUM_LEDS + steps  # Total frames for each sweep

        # Sweep from bottom to top
        for frame in range(total_frames):
            start_time = time.time()

            # Activate the next LED in the sweep
            if frame < NUM_LEDS:
                led_index = frame
                led_strip[led_index] = color

                # Start fading the previous LED
                if led_index > 0:
                    fading_leds[led_index - 1] = steps
            else:
                # After the last LED, ensure the final LED starts fading
                if (NUM_LEDS - 1) not in fading_leds:
                    fading_leds[NUM_LEDS - 1] = steps

            # Update fading LEDs
            for led in list(fading_leds.keys()):
                fade_step = fading_leds[led]
                fade_factor = fade_step / float(steps)
                r, g, b = color
                led_strip[led] = (
                    int(r * fade_factor),
                    int(g * fade_factor),
                    int(b * fade_factor)
                )
                fading_leds[led] -= 1
                if fading_leds[led] <= 0:
                    led_strip[led] = (0, 0, 0)  # Turn off the LED completely
                    del fading_leds[led]

            # Display the updated strip
            led_strip.show()

            # Calculate elapsed time and sleep to maintain frame rate
            elapsed = time.time() - start_time
            time_to_sleep = frame_duration - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            else:
                # If processing took longer than frame duration, skip sleep
                pass

        # Sweep from top to bottom
        fading_leds = {}  # Reset fading LEDs for the reverse sweep
        for frame in range(total_frames):
            start_time = time.time()

            # Activate the next LED in the reverse sweep
            if frame < NUM_LEDS:
                led_index = NUM_LEDS - 1 - frame
                led_strip[led_index] = color

                # Start fading the next LED
                if led_index < NUM_LEDS - 1:
                    fading_leds[led_index + 1] = steps
            else:
                # After the first LED, ensure the first LED starts fading
                if 0 not in fading_leds:
                    fading_leds[0] = steps

            # Update fading LEDs
            for led in list(fading_leds.keys()):
                fade_step = fading_leds[led]
                fade_factor = fade_step / float(steps)
                r, g, b = color
                led_strip[led] = (
                    int(r * fade_factor),
                    int(g * fade_factor),
                    int(b * fade_factor)
                )
                fading_leds[led] -= 1
                if fading_leds[led] <= 0:
                    led_strip[led] = (0, 0, 0)  # Turn off the LED completely
                    del fading_leds[led]

            # Display the updated strip
            led_strip.show()

            # Calculate elapsed time and sleep to maintain frame rate
            elapsed = time.time() - start_time
            time_to_sleep = frame_duration - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            else:
                # If processing took longer than frame duration, skip sleep
                pass

    # Turn off the strip after completion
    led_strip.fill((0, 0, 0))
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
