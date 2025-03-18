def process_csv(file_path, output_file, idx):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Check if timestamps are in milliseconds and convert if necessary
    if df['Timestamp'].max() > 100000:  # If values are too large, assume milliseconds
        df['Timestamp'] = df['Timestamp'] / 1000.0
    
    # Convert blink status to numerical values (0 = Not Blinking, 1 = Blinking)
    df['Left Eye Status'] = df['Left Eye Status'].apply(lambda x: 1 if x.strip().lower() == 'blinking' else 0)
    df['Right Eye Status'] = df['Right Eye Status'].apply(lambda x: 1 if x.strip().lower() == 'blinking' else 0)
    
    # Compute standard deviation of head pose (x, y, z)
    head_std_x = df['Head Pose X'].std()
    head_std_y = df['Head Pose Y'].std()
    head_std_z = df['Head Pose Z'].std()
    
    # Compute mean eye velocity (change in pupil position over time)
    dt = df['Timestamp'].diff().fillna(1)  # Avoid division by zero
    
    left_eye_velocity = np.sqrt(df['Left Pupil X'].diff()**2 + df['Left Pupil Y'].diff()**2) / dt
    right_eye_velocity = np.sqrt(df['Right Pupil X'].diff()**2 + df['Right Pupil Y'].diff()**2) / dt
    
    mean_left_eye_velocity = left_eye_velocity.mean()
    mean_right_eye_velocity = right_eye_velocity.mean()
    
    # Compute blink frequency (blinks per second)
    total_time = df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0] if len(df) > 1 else 1
    
    # Ensure total_time is not zero or NaN
    if total_time <= 0 or pd.isna(total_time):
        blink_frequency = 0
    else:
        blink_frequency = blinks / total_time #blinks is updated by the Gaze Tracking class
    
    # Create a DataFrame for computed values in a single row
    computed_data = pd.DataFrame([{
        'Id' : idx,
        'Head Pose X Std': head_std_x,
        'Head Pose Y Std': head_std_y,
        'Head Pose Z Std': head_std_z,
        'Mean Left Eye Velocity': mean_left_eye_velocity,
        'Mean Right Eye Velocity': mean_right_eye_velocity,
        'Blink Frequency': blink_frequency
    }])
    
    # Append computed values as a single row to the given CSV
    computed_data.to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)
    print(f"Computed values appended as a single row to: {output_file}")
