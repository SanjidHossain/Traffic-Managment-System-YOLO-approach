import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import os
from threading import Event
from collections import deque

# Load the YOLO model
model = YOLO('Models/bestV2.pt')

# Define the class names we are interested in (vehicles)
class_names = ['bicycle', 'bus', 'car', 'motorbike']  # Person is excluded from vehicle count

# Global variables for control
stop_event = Event()
pause_event = Event()

def draw_traffic_light(frame, state):
    height, width = frame.shape[:2]
    light_height = height // 3
    light_width = width // 10
    light = np.zeros((light_height, light_width, 3), dtype=np.uint8)

    if state == 'red':
        cv2.circle(light, (light_width//2, light_height//4), light_width//4, (255, 0, 0), -1)  # Red in BGR
    elif state == 'yellow':
        cv2.circle(light, (light_width//2, light_height//2), light_width//4, (255, 255, 0), -1)  # Yellow in BGR
    elif state == 'green':
        cv2.circle(light, (light_width//2, 3 * light_height // 4), light_width//4, (0, 255, 0), -1)  # Green in BGR

    frame[0:light_height, 0:light_width] = light
    return frame

def process_video(video_file, signal_state):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        temp_filename = tfile.name

    cap = cv2.VideoCapture(temp_filename)

    if not cap.isOpened():
        st.error(f"Error opening video file: {temp_filename}")
        return

    vehicle_count_buffer = deque(maxlen=30)  # Buffer to store vehicle counts for the last 30 frames

    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning of video
                continue

            # Perform object detection
            results = model(frame)

            # Process results and filter for vehicle classes
            vehicle_count = 0
            detections = results[0].boxes.data
            for detection in detections:
                x1, y1, x2, y2, score, class_id = detection

                # Only draw and count vehicles (bicycle, bus, car, motorbike)
                if class_id in [0, 1, 2, 3]:  # Indices for vehicle classes
                    vehicle_count += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Add the current frame's vehicle count to the buffer
            vehicle_count_buffer.append(vehicle_count)

            # Use the maximum vehicle count from the buffer
            max_vehicle_count = max(vehicle_count_buffer)

            # Draw traffic light
            frame = draw_traffic_light(frame, signal_state.value)

            # Convert colors from BGR to RGB for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame, max_vehicle_count

    finally:
        cap.release()
        try:
            os.unlink(temp_filename)
        except PermissionError:
            pass  # If we can't delete the file now, it will be deleted later

def main():
    st.title("Traffic Signal Management System")

    # Video file uploaders
    video_files = []
    for i in range(3):
        video_file = st.file_uploader(f"Upload video {i + 1}", type=['mp4', 'avi'])
        if video_file:
            video_files.append(video_file)

    if len(video_files) < 3:
        st.warning("Please upload at least 3 video files.")
        return

    # Red light duration slider
    red_light_duration = st.slider("Red Light Duration (seconds)", min_value=5, max_value=60, value=10)
    yellow_light_duration = st.slider("Yellow Light Duration (seconds)", min_value=1, max_value=5, value=1)
    max_green_light_duration = 30  # You can change this value to 60 or 120 or any desired limit

    # Initialize signal states
    class SignalState:
        def __init__(self):
            self.value = 'red'
            self.last_change = time.time()
            self.vehicle_count = 0
            self.paused_frame = None
            self.transition_count = 0  # Track the number of transitions without green

    signal_states = [SignalState() for _ in video_files]

    # Control buttons
    col1, col3 = st.columns(2)
    start_button = col1.button("Start Traffic Management")
    stop_button = col3.button("Stop")

    # Create placeholders for video streams and counters in a grid layout
    video_placeholders = []
    counter_placeholders = []

    # Arrange video feeds in a grid with titles
    for i in range(0, len(video_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(video_files):
                with col:
                    st.subheader(f"Road {i + j + 1}")
                    video_placeholders.append(st.empty())
                    counter_placeholders.append(st.empty())

    if start_button:
        stop_event.clear()
        pause_event.clear()

        video_generators = [process_video(vf, state) for vf, state in zip(video_files, signal_states)]
        last_frames = [next(gen) for gen in video_generators]

        # Determine the initial road with the most vehicles
        initial_vehicle_counts = [frame[1] for frame in last_frames]
        current_green = initial_vehicle_counts.index(max(initial_vehicle_counts))
        signal_states[current_green].value = 'green'
        signal_states[current_green].last_change = time.time()

        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.1)
                continue

            vehicle_counts = []

            for i, (gen, placeholder, counter, state) in enumerate(zip(video_generators, video_placeholders, counter_placeholders, signal_states)):
                if state.value != 'green':  # Red and yellow signals

                    if state.paused_frame is None:  # Capture the paused frame when red
                        state.paused_frame = last_frames[i][0]

                    # Perform object detection on paused frame
                    frame, vehicle_count = last_frames[i]

                    # Update vehicle count and bounding boxes based on paused frame
                    state.vehicle_count = vehicle_count

                    # Update counter
                    counter.text(f"Road {i + 1} - Vehicles: {vehicle_count}")
                    vehicle_counts.append(vehicle_count)

                    # Draw traffic light on the paused frame
                    frame = draw_traffic_light(frame, state.value)
                    placeholder.image(frame)

                else:  # Green signals keep playing the video normally
                    try:
                        frame, vehicle_count = next(gen)
                        last_frames[i] = (frame, vehicle_count)
                        state.paused_frame = None  # Reset paused frame

                        vehicle_counts.append(vehicle_count)  # Green doesn't update its vehicle count

                        # Draw traffic light
                        frame = draw_traffic_light(frame, state.value)
                        placeholder.image(frame)

                    except StopIteration:
                        st.warning(f"Video {i + 1} has ended. Restarting...")

            # Signal switching logic remains unchanged
            current_time = time.time()
            time_since_last_change = current_time - signal_states[current_green].last_change

            # Check if max green light duration has passed
            if signal_states[current_green].value == 'green' and time_since_last_change >= max_green_light_duration:
                # Turn green light to yellow
                signal_states[current_green].value = 'yellow'
                for _ in range(yellow_light_duration):
                    for i, (placeholder, state) in enumerate(zip(video_placeholders, signal_states)):
                        frame, _ = last_frames[i]
                        frame = draw_traffic_light(frame, state.value)
                        placeholder.image(frame)
                    time.sleep(1)

                # Turn yellow light to red
                signal_states[current_green].value = 'red'
                signal_states[current_green].last_change = current_time
                current_green = (current_green + 1) % len(video_files)  # Move to the next road in line

            elif time_since_last_change >= red_light_duration:
                # Increment transition count for all red roads
                for state in signal_states:
                    if state.value == 'red':
                        state.transition_count += 1

                # Check for roads that have not received green light for 4 transitions
                priority_roads = [i for i, state in enumerate(signal_states) if state.transition_count >= 4]

                if priority_roads:
                    # Sort priority roads by vehicle count
                    priority_roads.sort(key=lambda i: signal_states[i].vehicle_count, reverse=True)
                    next_green = priority_roads[0]
                else:
                    # Compare vehicle counts of red roads and switch the most crowded to green
                    red_roads = [i for i, state in enumerate(signal_states) if state.value == 'red']
                    if red_roads:
                        next_green = max(red_roads, key=lambda i: signal_states[i].vehicle_count)

                # Set the current green signal to yellow
                signal_states[current_green].value = 'yellow'

                # Show yellow light for the specified duration
                for _ in range(yellow_light_duration):
                    for i, (placeholder, state) in enumerate(zip(video_placeholders, signal_states)):
                        frame, _ = last_frames[i]
                        frame = draw_traffic_light(frame, state.value)
                        placeholder.image(frame)
                    time.sleep(1)

                # Switch the current yellow to red
                signal_states[current_green].value = 'red'

                # Set the new green signal to yellow first
                signal_states[next_green].value = 'yellow'
                for _ in range(yellow_light_duration):
                    for i, (placeholder, state) in enumerate(zip(video_placeholders, signal_states)):
                        frame, _ = last_frames[i]
                        frame = draw_traffic_light(frame, state.value)
                        placeholder.image(frame)
                    time.sleep(1)

                # Finally, switch the new yellow to green
                current_green = next_green
                signal_states[current_green].value = 'green'
                signal_states[current_green].last_change = time.time()
                signal_states[current_green].transition_count = 0  # Reset transition count for the new green road

    if stop_button:
        stop_event.set()

if __name__ == "__main__":
    main()