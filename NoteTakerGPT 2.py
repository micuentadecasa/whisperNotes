import requests
import openai
import traceback
import pandas as pd
import numpy as np
import tiktoken
import pyaudio
import wave
import keyboard
import time
import threading
import multiprocessing
from multiprocessing import Manager
from queue import Queue
import os

# Initialize OpenAI API key
openai.api_key = "OPENAI_API_KEY"

import os
import dotenv

# Set up API keys
dotenv.load_dotenv() 

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
#os.environ["OPENAI_API_KEY"] = OPENAI_KEY

openai.api_key = OPENAI_KEY

# Constants
MODEL = "gpt-4"
RECORDING_INTERVAL_SECONDS = 20

def split_text_into_chunks(text, chunk_size=20):
    sentences = text.split('\n')
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    return ['. '.join(chunk) for chunk in chunks]

def passage_segmenter(passage, interval=600):
    segment = []
    count = 0
    while count < len(passage):
        segment.append(passage[count:count + interval])
        count += interval
    return segment

import openai

from openai import OpenAI

def ask_question(messages):
    print(messages)
    client = OpenAI()

    stream = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
    # return all the responses
    responses = []
    for chunk in stream:
        responses.append(chunk.choices[0].text)
    return responses
    


    response = openai.Completion.create(
        model="gpt-4",  # Specify the GPT-4 model or any other available models
        messages=messages,
        max_tokens=1500  # Set a suitable limit for the number of tokens in the response
    )
    # Extracting content based on the structure of the response
    return response['choices'][0]['message']['content'] if 'content' in response['choices'][0]['message'] else response['choices'][0]['text']


def count_tokens(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def create_analysis(passage):
    print("Creating analysis for passage: " + passage)
    messages = [
      {"role": "system", "content": "Read the following section from a meeting transcript and generate detailed bullet point notes on all the key points and details being discussed.\n\nSection: " + passage}
    ]
    return ask_question(messages)

def intermediate_notes(passage):
    messages = [
      {"role": "system", "content": """Read the following collection of bullet point notes on a meeting and rewrite them to improve flow and make space for additional entities with fusion, compression, and removal of uninformative phrases like \"the meeting participants are discussing\". The new section of bullet pointed notes that you make should become highly dense and concise yet self-contained, i.e., easily understood, even for people who didnt attend the meeting. They must also still be in bullet point format.\n\nSection: """ + passage}
    ]
    return ask_question(messages)

def reformat_analysis(analysis, seg_count):
    tok_len = count_tokens(analysis)
    if tok_len <=6000:
        messages = [
            {"role": "system", "content": "You are a helpful, GPT-4 powered sumnmary generator. You will be given a set of bullet point notes broken out into " + seg_count + " sections, each corresponding to a section of a call transctipt and you will consolidate all the sections into one single very detailed, very comprehensive set of bullet point notes organized by topic rather than section. Your outputted notes should be a minimum of 2300 words in length, so make sure to capture all the details without watering them down, shortening them, nor skipping any"},
            {"role": "user", "content": ".\n\nNotes: " + analysis}
        ]
        return ask_question(messages)
    else:
        meeting_notes = analysis
        num_segments = seg_count
        notes_tok_len = tok_len
        while notes_tok_len > 6000:
            shortened_notes = ""
            segments = split_text_into_chunks(meeting_notes)
            for segment in segments:
                new_segment = intermediate_notes(segment)
                shortened_notes += new_segment + "\n"
            meeting_notes = shortened_notes
            num_segments = len(segments)
            notes_tok_len = count_tokens(meeting_notes)
        messages = [
            {"role": "system", "content": "You are a helpful, GPT-4 powered sumnmary generator. You will be given a set of bullet point notes broken out into " + num_segments + " sections, each corresponding to a section of a call transctipt and you will consolidate all the sections into one single very detailed, very comprehensive set of bullet point notes organized by topic rather than section. Your outputted notes should be a minimum of 2300 words in length, so make sure to capture all the details without watering them down, shortening them, nor skipping any"},
            {"role": "user", "content": ".\n\nNotes: " + meeting_notes}
        ]
        return ask_question(messages)

import whisper
import numpy as np

model = whisper.load_model("tiny")  # Load the Whisper model, select appropriate size
   

import resampy
import soundfile as sf

def resample_audio(input_file, output_file, target_sr=16000):
    # Load the audio file
    data, sr = sf.read(input_file)

    # Check if the sampling rate is already 16 kHz
    if sr != target_sr:
        # Resample the audio to 16 kHz
        data_resampled = resampy.resample(data, sr, target_sr)
        # Save the resampled audio
        sf.write(output_file, data_resampled, target_sr)
        print(f"Resampled audio saved to {output_file}")
    else:
        print(f"Audio is already at {target_sr} Hz")

import subprocess

def transcribe_to_txt(input_filename: str, output_filename: str):
    # let's create a tmp file changing the name of hte input file
    # so we don't overwrite the original file
    tmp_filename = "new" + input_filename
    #let's resample the file
    resample_audio(input_filename, tmp_filename)


    print('Running whisper transcription...')
    # Compose the command of all components
    command = ['./main', '-f', tmp_filename, '-otxt','-l','auto', '-of', output_filename, '-np']

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    print ("finised transcribing")
    # Print the transcribed text
    with open(output_filename + ".txt" , 'r') as file:
        print(file.read())
    # Clean up temporary files
    os.remove(output_filename  + ".txt")
    os.remove(tmp_filename )


def transcribe_audio(audio_queue, result_queue, shared_list):
    print ("Transcribing audio")
    while True:
        audio_file = audio_queue.get()
        if audio_file is None:
            break
        # let's use main.cpp for transcribing audio
            
        # Prepare the output filename
        output_filename = audio_file.replace('.wav', '')
        
        # Transcribe the audio to text using our whisper.cpp wrapper
        transcribe_to_txt(audio_file, output_filename)


def record_audio(filename, rate=44100, channels=1, chunk=2048, format=pyaudio.paInt16, stop_event=None):
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk,
                    input_device_index=2)

    frames = []
    start_time = time.time()
    recording_started = False

    while not stop_event.is_set():
        if not recording_started:
            recording_started = True
            start_time = time.time()

        try:
            data = stream.read(chunk)
            frames.append(data)
        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                data = '\x00' * chunk  # Return a silent block if an overflow occurs
                frames.append(data)

        if time.time() - start_time >= RECORDING_INTERVAL_SECONDS:
            break

    try:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"An error occurred while stopping the stream: {e}")
    

    p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def main():
    # Create a multiprocessing Event to signal recording termination
    stop_event = multiprocessing.Event()
    # Create a multiprocessing Queue for transcription results
    result_queue = multiprocessing.Queue()
    # Create a Manager object and a shared list
    manager = Manager()
    shared_list = manager.list()

    # Start the transcription process in parallel
    transcription_process = multiprocessing.Process(
        target=transcribe_audio, args=(result_queue, result_queue, shared_list)  # Pass result_queue twice
    )
    transcription_process.start()
    print("Process has started")
    counter = 1
    try:
        while True:
            filename = f"recorded_audio{counter}.wav"
            # Start recording immediately
            record_audio(filename, stop_event=stop_event)
            # Queue the audio file for transcription
            result_queue.put(filename)
            counter += 1
    except KeyboardInterrupt:
        # Stop recording and transcription when Ctrl+C is pressed
        stop_event.set()

    transcription_process.join()
    # At this point, all transcriptions have gone through the queue
    unformatted = "\n".join(shared_list)  # Join all the elements in the shared list
    # Perform the reformatting after all transcriptions are done
    notes = reformat_analysis(unformatted, str(counter - 1))
    print("\n\n\n\n")
    print(notes)

if __name__ == "__main__":
    main()