# Data Simulator
A Streamlit web application for generating realistic simulated data based on statistical analysis of existing datasets. Features automatic file format detection and robust error handling.
Features
Support for multiple file formats:

.xlsx, .csv & .txt

Statistical distribution detection
Seasonality pattern preservation
Realistic noise generation
Interactive date range selection
Visual comparison between original and simulated data
Multiple export formats (CSV and Excel)
Robust error handling and fallback options

## Installation
1. Clone the repository:
```bash
git clone https://github.com/6ogo/dataset-generator.git
cd dataset-generator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the Streamlit app:
```bash
streamlit run dataGenerator.py
```
2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)
3. Upload your Excel file containing the original dataset
4. Select the column you want to simulate
5. Choose the date range for the simulated data
6. Click "Generate Simulated Data" to create and visualize the new dataset
7. Download the simulated data in CSV format

## Requirements
Create a `requirements.txt` file with the following dependencies:

```
streamlit
pandas
numpy
scipy
matplotlib
```

## Input Data Format
The application expects:
- Excel files (.xlsx)
- At least one numeric column for simulation
- Optional date column for seasonality detection

## How It Works
1. **Data Analysis**: The app analyzes your input data to determine:
   - Statistical distribution
   - Seasonality patterns
   - Key statistical parameters

2. **Simulation Process**:
   - Generates base values following the detected distribution
   - Applies seasonal patterns if detected
   - Adds realistic noise
   - Ensures values stay within reasonable bounds

3. **Visualization**:
   - Displays histograms comparing original and simulated data
   - Shows basic statistics of the dataset

## Contributing
Feel free to submit issues and enhancement requests!

## License
[MIT](https://choosealicense.com/licenses/mit/)