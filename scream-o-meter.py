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
BRIGHTNESS = 1.0  # Scale 0-1

# PyAudio constants
SAMPLE_RATE = 44100
CHANNELS = 2  # Use mono input to simplify processing
BUFFER_SIZE = 1024  # Size of buffer for PyAudio

# Audio processing constants
NOISE_THRESHOLD = 2000.0  # Noise floor to filter out background noise
SCALING_FACTOR = 10.0  # Logarithmic scaling factor
ALPHA = 0.4  # Smoothing factor (increased for better smoothing)

# High score tracking
INACTIVITY_PERIOD = 10  # Seconds to reset session high score
HIGH_SCORE_RESET = 60  # Seconds to reset overall high score

pyaudio_instance = pyaudio.PyAudio()

class LEDController:
    def __init__(self):
        # Initialize your variables
        self.num_leds = 144  # Example number of LEDs
        self.led_strip = neopixel.NeoPixel(LED_PIN, NUM_LEDS, auto_write=False)  # Initialize your LED strip object
        self.color = (255, 131, 0)  # Initial color
        self.current_pixel = 0  # Start at pixel 0
        self.previous_pixel = 0  # Keep track of the previous pixel
        self.active_pixels = {}  # Dictionary to hold active pixels with brightness and color
        self.max_brightness = 255  # Maximum brightness
        self.fade_step = 10  # Amount by which brightness decreases each update
        self.movement_speed = 3  # Number of pixels to move each update
        self.trail_direction = 1  # Initial direction
        self.smoothed_display_value = 0
        self.alpha = 0.1  # Smoothing factor
        self.current_zone = 'orange'
        # Define color zones with hysteresis thresholds
        self.color_zones = [
            {
                'name': 'green',
                'color': (0, 255, 0),
                'upper_threshold': self.num_leds * 0.95,
                'lower_threshold': self.num_leds * 0.95
            },
            {
                'name': 'red',
                'color': (255, 0, 0),
                'upper_threshold': self.num_leds * 0.80,
                'lower_threshold': self.num_leds * 0.80
            },
            {
                'name': 'magenta',
                'color': (255, 0, 255),
                'upper_threshold': self.num_leds * 0.50,
                'lower_threshold': self.num_leds * 0.50
            },
            {
                'name': 'orange',
                'color': (255, 131, 0),
                'upper_threshold': 0,
                'lower_threshold': 0
            }
        ]

    def update_leds(self, display_value, high_score, session_high_score):
        # Update the color based on the current display value
        new_zone = self.current_zone  # Default to the current zone
        for zone in self.color_zones:
            if self.current_zone == zone['name']:
                # Check if we need to switch to a lower zone
                if self.current_pixel < zone['lower_threshold']:
                    continue  # Continue checking lower zones
                else:
                    new_zone = zone['name']
                    break  # Stay in the current zone
            else:
                # Check if we need to switch to a higher zone
                if self.current_pixel >= zone['upper_threshold']:
                    new_zone = zone['name']
                    break
        if new_zone != self.current_zone:
            self.current_zone = new_zone
            self.color = next(zone['color'] for zone in self.color_zones if zone['name'] == new_zone)
        
        target_pixel = int(display_value)

        # Determine direction to move
        if target_pixel > self.current_pixel:
            self.trail_direction = 1
        elif target_pixel < self.current_pixel:
            self.trail_direction = -1
        else:
            self.trail_direction = 0

        # Keep track of the previous position
        self.previous_pixel = self.current_pixel

        # Move current_pixel towards target_pixel
        if self.trail_direction != 0:
            self.current_pixel += self.trail_direction * self.movement_speed
            # Ensure current_pixel is within bounds
            self.current_pixel = max(0, min(self.current_pixel, self.num_leds - 1))
            # Determine the range of pixels to light up
            if self.trail_direction > 0:
                pixel_range = range(self.previous_pixel + 1, self.current_pixel + 1)
            else:
                pixel_range = range(self.previous_pixel - 1, self.current_pixel - 1, -1)
            # Add intermediate pixels to active_pixels
            for pixel in pixel_range:
                self.active_pixels[pixel] = {'brightness':self.max_brightness, 'color':self.color}
        else:
            # If there's no movement, ensure the current pixel is added
            self.active_pixels[self.current_pixel] = self.max_brightness

        # Ensure current_pixel is within bounds
        self.current_pixel = max(0, min(self.current_pixel, self.num_leds - 1))

        # Add current_pixel to active_pixels with max brightness
        self.active_pixels[self.current_pixel] = {'brightness':self.max_brightness, 'color':self.color}

        # Clear all LEDs
        self.led_strip.fill((0, 0, 0))

        # Update and display active pixels
        pixels_to_remove = []
        for pixel in list(self.active_pixels.keys()):
            pixel_info = self.active_pixels[pixel]
            brightness = pixel_info['brightness']
            color = pixel_info['color']
            # Compute color based on brightness
            r, g, b = color
            factor = brightness / self.max_brightness
            self.led_strip[pixel] = (
                int(r * factor),
                int(g * factor),
                int(b * factor),
            )
            # Decrease brightness
            brightness -= self.fade_step
            if brightness <= 0:
                pixels_to_remove.append(pixel)
            else:
                self.active_pixels[pixel]['brightness'] = brightness  # Update brightness

        # Remove pixels that have faded out
        for pixel in pixels_to_remove:
            del self.active_pixels[pixel]

        # Display the updated strip
        self.led_strip.show()

        # Optional: Add a small delay to control the speed
        time.sleep(0.001)

    def clear(self):
        self.led_strip.fill((0, 0, 0))
        self.led_strip.show()
        
    def __exit__(self):
        self.clear()

class AudioProcessor:
    def __init__(self):
        self.stream_in = pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=1,
            frames_per_buffer=BUFFER_SIZE,
        )
        self.smoothed_value = 0.3
        self.high_score = 0
        self.session_high_score = 0
        self.last_high_score_time = time.time()

    def apply_logarithmic_scaling(self, input_value, max_value, scaling_factor):
        # Ensure input_value is at least 1 to avoid math domain error
        input_value = max(input_value, 1.0)
        # Calculate the logarithm of the input value
        log_value = math.log(input_value) / math.log(scaling_factor)
        # Scale the logarithmic value to the max_value range
        scaled_value = (log_value / (math.log(32767) / math.log(scaling_factor))) * max_value
        return min(max_value, scaled_value)

    def process_audio(self):
        data = self.stream_in.read(BUFFER_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Calculate RMS (Root Mean Square) to get a better measure of loudness
        mean = np.mean(np.abs(audio_data))

        # Apply noise threshold
        if mean < NOISE_THRESHOLD:
            mean = 0.0

        # Apply logarithmic scaling
        mapped_value = self.apply_logarithmic_scaling(mean, NUM_LEDS, SCALING_FACTOR)

        # Exponential smoothing
        self.smoothed_value = (ALPHA * mapped_value) + ((1 - ALPHA) * self.smoothed_value)
        display_value = int(self.smoothed_value)

        # Update high scores
        current_time = time.time()
        if display_value > self.high_score:
            self.high_score = display_value

        if display_value > self.session_high_score:
            self.session_high_score = display_value
            self.last_high_score_time = current_time

        # Reset session high score after inactivity
        if current_time - self.last_high_score_time > INACTIVITY_PERIOD:
            self.session_high_score = 0

        return display_value, self.high_score, self.session_high_score

    def close(self):
        self.stream_in.stop_stream()
        self.stream_in.close()
        pyaudio_instance.terminate()

class MotionSensor:
    def __init__(self):
        self.pir = digitalio.DigitalInOut(board.D13)
        self.pir.direction = digitalio.Direction.INPUT

    def detect_motion(self):
        return self.pir.value

class AudioPlayer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio_data, self.sample_rate = sf.read(self.file_path, dtype='float32')
        self.current_frame = 0
        self.stream_out = None

    def play(self):
        self.stream_out = pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=self.audio_data.shape[1] if len(self.audio_data.shape) > 1 else 1,
            rate=self.sample_rate,
            output=True,
            output_device_index=3,
            stream_callback=self.callback,
        )
        self.stream_out.start_stream()

        while self.stream_out.is_active():
            time.sleep(0.1)

        self.stream_out.stop_stream()
        self.stream_out.close()

    def callback(self, out_data, frame_count, time_info, status):
        end = self.current_frame + frame_count
        data_chunk = self.audio_data[self.current_frame:end]
        self.current_frame = end
        if len(data_chunk) < frame_count:
            return (data_chunk.tobytes(), pyaudio.paComplete)
        else:
            return (data_chunk.tobytes(), pyaudio.paContinue)

def scream_o_meter():
    led_controller = LEDController()
    audio_processor = AudioProcessor()

    try:
        while True:
            display_value, high_score, session_high_score = audio_processor.process_audio()
            led_controller.update_leds(display_value, high_score, session_high_score)
            time.sleep(0.01)  # Adjust as needed
    except KeyboardInterrupt:
        pass
    finally:
        led_controller.clear()
        audio_processor.close()

def motion_sense():
    motion_sensor = MotionSensor()
    while True:
        if motion_sensor.detect_motion():
            print("Motion Detected")
            audio_player = AudioPlayer("../test_recording.wav")
            audio_player.play()
            time.sleep(5)  # Debounce time
        time.sleep(0.1)  # Polling interval

def main():
    motion_thread = threading.Thread(target=motion_sense, daemon=True)
    scream_thread = threading.Thread(target=scream_o_meter, daemon=True)

    motion_thread.start()
    scream_thread.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Threads will be closed automatically on exit
        pass

if __name__ == "__main__":
    main()
