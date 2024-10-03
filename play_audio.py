import pyaudio
import soundfile as sf

file="../test_recording.wav"

data, sample_rate = sf.read(file, dtype='float32')


p = pyaudio.PyAudio()


def callback(in_data, frame_count, time_info, status):
    global current_frame
    end = current_frame + frame_count

    # Check if we're at the end of the audio file
    if end > len(data):
        data_chunk = data[current_frame:]  # Get remaining data
        return (data_chunk.tobytes(), pyaudio.paComplete)  # Signal end of stream

    # If we have enough data, return the chunk
    data_chunk = data[current_frame:end]
    current_frame = end
    return (data_chunk.tobytes(), pyaudio.paContinue)

stream = p.open(format=pyaudio.paFloat32,
                channels=data.shape[1] if len(data.shape) > 1 else 1,
                rate=sample_rate,
                output=True,
                output_device_index=1,
                stream_callback=callback)

def play():
    stream.start_stream()

    while stream.is_active():
        pass
    stream.stop_stream()


current_frame = 0
play()


stream.close()

p.terminate()