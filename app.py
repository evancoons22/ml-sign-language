from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin  # Import CORS
import sqlite3
from sqlite3 import Error

import torch
from flask import Flask, request, jsonify
from hand_pose_model import HandPoseGNN
from torch_geometric.data import Data
import numpy as np

app = Flask(__name__)
CORS(app, origins='http://localhost:8000')  # Enable CORS for all routes
DATABASE = 'hand_poses.db'


def create_connection(db_file):
    """ Create a database connection to the SQLite database specified by db_file """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

def create_table():
    """ Create table for hand pose data """
    conn = create_connection(DATABASE)
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS hand_pose_data (
                             id INTEGER PRIMARY KEY,
                             label TEXT,
                             {});'''.format(
                ', '.join([f'x{k} REAL, y{k} REAL' for k in range(21)])
            ))
        except Error as e:
            print(e)
        finally:
            conn.close()

@app.route('/receive_data', methods=['POST'])
@cross_origin()
def receive_data():
    """ Endpoint to receive hand pose data """
    
    request.get_json(force=True)
    data = request.json
    label = data.get('label')
    hand_data = data.get('handData')
    keypoints = hand_data[0].get('keypoints')

    # print("label: ", keypoints)

    # Flatten hand data
    # hand_points = [item for sublist in hand_data for item in sublist['keypoints']]
    # values = [label] + hand_points

    #flatten
    flattened_keypoints = [coord for point in keypoints for coord in (point['x'], point['y'])]

    # Add label to the data
    label = data.get('label', 'No Label')
    values = [label] + flattened_keypoints

    print("logging data for prediction: ", label)

    # print("logging data: ", values)

    # Insert data into the database
    conn = create_connection(DATABASE)
    if conn is not None:
        try:
            c = conn.cursor()
            c.execute('INSERT INTO hand_pose_data (label, {}) VALUES ({});'.format(
                ', '.join([f'x{k}, y{k}' for k in range(21)]),
                ', '.join(['?'] * 43)
            ), values)
            conn.commit()
        except Error as e:
            print(e)
        finally:
            conn.close()

    return jsonify({'status': 'success'})


model = HandPoseGNN(10)  # Replace with your actual model and parameters
model.load_state_dict(torch.load('model.pth'))
model.eval()

edge_list = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (0,18), (0,19), (0,20)] # Example edges
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

    request.get_json(force=True)
    data = request.json
    #label = data.get('label')
    hand_data = data.get('handData')
    keypoints = hand_data[0].get('keypoints')

    # print("label: ", keypoints)

    # Flatten hand data
    # hand_points = [item for sublist in hand_data for item in sublist['keypoints']]
    # values = [label] + hand_points

    #flatten
    values = [coord for point in keypoints for coord in (point['x'], point['y'])]
    # values = values[1:]

    keypoints = np.array(values).reshape(-1, 2) # Reshape to (21, 2)
    print("keypoints: ", keypoints)
    # Normalize keypoints (optional, based on your data range)
    keypoints = (keypoints - keypoints.mean(axis=0)) / keypoints.std(axis=0)
    x = torch.tensor(keypoints, dtype=torch.float)
    input = Data(x=x, edge_index=edge_index)

    # Add label to the data
    # print("logging data: ", values)
    # Parse input data
    # data = request.json
    # Convert data to suitable format for PyTorch model
    # For example, if data is a sequence of hand positions:
    # input_tensor = torch.tensor(values)  # Adjust based on actual data format

    # Forward pass
    with torch.no_grad():
        output = model(input)
        _, predicted = torch.max(output.data, 1)

    # Convert the prediction to a response
    response = {
        'prediction': predicted.item()
    }
    print("response: ", response)
    return jsonify(response)

if __name__ == '__main__':
    create_table()
    app.run(debug=True)

