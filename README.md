# Scania Component X Project

This project utilizes the Scania Component X dataset to analyze and visualize operational readouts, specifications, and time-to-event data related to vehicle performance. The dataset consists of three main CSV files:

- **train_operational_readouts.csv**: Contains operational readouts data, including various metrics related to vehicle performance.
- **train_specifications.csv**: Details the specifications of the Scania Component X, outlining the characteristics and features of the components.
- **train_tte.csv**: Provides time-to-event data, which can be used for survival analysis or predictive modeling related to the components.

## Project Structure

```
scania-component-x-project
├── data
│   ├── train_operational_readouts.csv
│   ├── train_specifications.csv
│   └── train_tte.csv
├── src
│   ├── main.py
│   ├── data_loader.py
│   ├── analysis.py
│   └── utils.py
├── notebooks
│   └── exploration.ipynb
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd scania-component-x-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- To run the main application, execute:
  ```
  python src/main.py
  ```

- For exploratory data analysis, open the Jupyter notebook:
  ```
  jupyter notebook notebooks/exploration.ipynb
  ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.