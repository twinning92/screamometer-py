import pyaudio

p = pyaudio.PyAudio()

# List all available devices
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print(f"Device Index: {i}")
    print(f"  Name: {device_info['name']}")
    print(f"  Max Input Channels: {device_info['maxInputChannels']}")
    print(f"  Max Output Channels: {device_info['maxOutputChannels']}")
    print(f"Supported Sample Rates: {device_info['defaultSampleRate']}")

    print()

p.terminate()
