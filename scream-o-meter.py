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
ALPHA = 0.1  # Smoothing factor (increased for better smoothing)

# High score tracking
INACTIVITY_PERIOD = 10  # Seconds to reset session high score
HIGH_SCORE_RESET = 60  # Seconds to reset overall high score

pyaudio_instance = pyaudio.PyAudio()

class LEDController:
    def __init__(self):
        self.num_leds = 144  # Example number of LEDs
        self.led_strip = neopixel.NeoPixel(LED_PIN, NUM_LEDS, auto_write=False)  # Initialize your LED strip object
        self.current_pixel = 0  # Start at pixel 0
        self.previous_pixel = 0  # Keep track of the previous pixel
        self.active_pixels = {}  # Dictionary to hold active pixels and their brightness
        self.max_brightness = 255  # Maximum brightness
        self.fade_step = 20  # Amount by which brightness decreases each update
        self.movement_speed = 2  # Number of pixels to move each update
        self.trail_direction = 1  # Initial direction
        self.position_zones = [
            {
                'name': 'red',
                'color': (255, 0, 0),
                'start_pixel': int(self.num_leds * 0.95),
                'end_pixel': self.num_leds - 1
            },
            {
                'name': 'pink',
                'color': (255, 25, 20),
                'start_pixel': int(self.num_leds * 0.8),
                'end_pixel': int(self.num_leds * 0.95) - 1
            },
            {
                'name': 'magenta',
                'color': (100, 0, 255),
                'start_pixel': int(self.num_leds * 0.5),
                'end_pixel': int(self.num_leds * 0.8) - 1
            },
            {
                'name': 'orange',
                'color': (255, 25, 0),
                'start_pixel': 0,
                'end_pixel': int(self.num_leds * 0.5) - 1
            }
        ]

    def update_leds(self, display_value, high_score, session_high_score):
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
                pixel_color = self.get_color_for_pixel(pixel)  # Get color based on pixel position
                self.active_pixels[pixel] = {'brightness': self.max_brightness, 'color': pixel_color}
        else:
            # If there's no movement, ensure the current pixel is added
            pixel_color = self.get_color_for_pixel(self.current_pixel)  # Get color based on pixel position
            self.active_pixels[self.current_pixel] = {'brightness': self.max_brightness, 'color': pixel_color}
        
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
        
        # Adjust time delay for animation speed
        time.sleep(0.005)  # Adjust as needed
    
    def get_color_for_pixel(self, pixel_position):
        for zone in self.position_zones:
            if zone['start_pixel'] <= pixel_position <= zone['end_pixel']:
                return zone['color']
        # If no zone matches, return a default color (should not occur)
        return (255, 255, 255)  # White
    
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
        self.maximum_loudness = 0


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
        rms_value = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

        # Apply noise threshold
        if rms_value < NOISE_THRESHOLD:
            rms_value = 0.0
        if rms_value > self.maximum_loudness:
            self.maximum_loudness = rms_value
            print(f"max loudness: {self.maximum_loudness}")
        # Apply logarithmic scaling
        mapped_value = self.apply_logarithmic_scaling(rms_value, NUM_LEDS, SCALING_FACTOR)

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
        self.stream_out = pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=self.audio_data.shape[1] if len(self.audio_data.shape) > 1 else 1,
            rate=SAMPLE_RATE,
            output=True,
            output_device_index=1,
            stream_callback=self.callback,
        )

    def play(self):
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
