import spidev
import time
import numpy as np
import rpi_ws281x as ws
import board
import neopixel

# Configure SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # Open SPI bus 0, device 0
spi.max_speed_hz = 800000  # Set SPI speed (800 kHz for WS2812B)

# Constants for NeoPixel
NUM_LEDS = 144
BRIGHTNESS = 0.1

# Initialize the rpi_ws281x LED strip using SPI
led_strip = ws.PixelStrip(NUM_LEDS, spi.xfer2, brightness=BRIGHTNESS, strip_type=ws.WS2812_STRIP)
led_strip.begin()  # Initialize the strip

# Function to update LEDs
def update_leds(display_value, high_score, session_high_score):
    led_strip.fill((0, 0, 0))  # Clear the LED strip

    # Set LEDs up to the display value to blue
    for i in range(display_value):
        led_strip.setPixelColor(i, ws.Color(255, 104, 0))  # Set color (RGB)

    # Set session high score LED to green
    if session_high_score < NUM_LEDS:
        led_strip.setPixelColor(session_high_score, ws.Color(0, 255, 0))  # Green

    # Set high score LED to red
    if high_score < NUM_LEDS:
        led_strip.setPixelColor(high_score, ws.Color(255, 0, 0))  # Red
        led_strip.setPixelColor(high_score + 1, ws.Color(255, 0, 0))  # Red

    led_strip.show()  # Push the changes via SPI
