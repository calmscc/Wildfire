<div id="top"></div>

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

<br />
<div align="center">
  <a href="https://github.com/calmscc">
    <img src="https://img.icons8.com/external-flat-wichaiwi/64/undefined/external-bush-fire-climate-change-flat-wichaiwi.png" alt="Logo" width="80" height="80"/> 
  </a>

<h3 align="center">California Wildfire Risk Predictor</h3>

  <p align="center">
    Machine Learning Project
    <br/>
    <a href="https://github.com/calmscc/Wildfire"><strong>Explore the Repo »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/calmscc/Wildfire/blob/main/app.py">View Streamlit app code</a>
    ·
    <a href="https://github.com/calmscc/Wildfire/blob/main/notebooks/Wildfire_Model_Building.ipynb">Model Building Notebook</a>
    ·
    <a href="https://github.com/calmscc/Wildfire/blob/main/notebooks/Wildfire_EDA.ipynb">EDA on Wildfire Dataset</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

Predict wildfire risk in California using real environmental and weather data. This project uses a machine learning model deployed with a Streamlit app for interactive risk prediction based on user input and visualization of feature correlations.

* Train and use a model to estimate wildfire risk based on local weather variables.
* Display a correlation heatmap for interpretability.
* All code, models, and data are included for reproducibility.

## Deployed App

> [LINK TO LIVE APP](https://wildfire-u9ob.onrender.com/)


## How It Works

1. **User Input:** Enter weather/environment info in the web app.
2. **Feature Engineering:** Computes features like `TEMP_DIFF` automatically.
3. **Preprocessing:** Data is scaled and encoded to match the model's training.
4. **Prediction:** ML model predicts wildfire risk probability and assigns a risk level (Low / Moderate / High).
5. **Visualization:** View the feature correlation matrix as a heatmap.

## Example Usage

1. Input parameters (rain, temperature, wind, month, etc.)
2. Click "Predict Wildfire Risk" for probability & risk class.
3. Check "Show correlation heatmap" for feature insights.

---

### Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-%230080BA.svg?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/seaborn-%230475A8.svg?style=for-the-badge&logo=seaborn&logoColor=white)
![joblib](https://img.shields.io/badge/joblib-blue?style=for-the-badge)
![geopy](https://img.shields.io/badge/geopy-green?style=for-the-badge)

### Tools Used

![VSCode](https://img.shields.io/badge/VS%20Code-0078d7?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Render](https://img.shields.io/badge/Render-3A3550?style=for-the-badge&logo=render&logoColor=white)

---

## Contact

[![calmscc | GitHub](https://img.shields.io/badge/calmscc-eeeeee?style=for-the-badge&logo=github&logoColor=ffffff&labelColor=0A66C2)](https://github.com/calmscc)

---

## Acknowledgements

- Data: [CAL FIRE](https://gis.data.ca.gov/datasets/CALFIRE-Forestry::california-fire-perimeters-all/explore), [Zenodo Wildfire Dataset](https://zenodo.org/records/14712845)
- Machine Learning tips: scikit-learn documentation, Streamlit docs  
- README template and examples inspired by [aravind-selvam/forest-fire-prediction](https://github.com/aravind9722/Forest-fire_Prediction) and [catiaspsilva/README-template](https://github.com/catiaspsilva/README-template)[2].

---



<p align="right">(<a href="#top">back to top</a>)</p>


