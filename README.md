# Heritige Housing
Welcome to Heritige Housing, this project is to help a good friend with analysing homes in Ames Iowa.
# Contents
1. [Design](#design)
2. [User Stories](#user-stories)
3. [Dataset Content](#dataset-content)
4. [Business Requirements](#business-requirements)
5. [Hypothesis and validations](#hypothesis-and-how-to-validate)
6. [ML business case](#machine-learning-business-case)
7. [Dashboard Design](#dashboard-design)
8. [Deployment](#deployment)
9. [Unfixed Bugs](#unfixed-bugs)
10. [Fixed Bugs](#fixed-bugs)
11. [Testing](#testing)
12. [contributing](#contributing)
13. [Media](#media)
14. [Credits](#credits)
15. [Acknowledgements](#acknowledgements-optional)

## Note
 **Please Note that i am not happy with what i have done for this project and given more time and focus, i would make sure to fully test the application but due to a couple of personal commitments and family/mental health issues, the application isn't 100% complete its more like 85% complete.**
## Design

The design of this project follows a structured approach that aligns with the CRISP-DM framework (Cross-Industry Standard Process for Data Mining). It ensures that the solution is tailored to the business requirements and supports user goals through data analysis, visualization, and predictive modeling.

### Design Objectives
1. **Understand Key Factors Influencing House Prices**:
   - Provide visualizations and insights into the factors affecting house sale prices in Ames, Iowa.

2. **Accurate Price Prediction for Real Estate**:
   - Develop a predictive model to estimate the sale price of properties accurately based on various features.

### Design Approach
1. **Data Understanding**:
   - Explore and clean the dataset to understand key attributes influencing house prices.
   - Perform exploratory data analysis (EDA) to identify relationships and patterns.

2. **Data Preparation**:
   - Handle missing values and outliers to ensure the quality of the data.
   - Feature engineering to transform raw data into meaningful inputs for the predictive model.

3. **Modeling**:
   - Train multiple regression models to find the best-performing algorithm for predicting house prices.
   - Use feature importance analysis to identify the most influential factors.

4. **Evaluation**:
   - Assess model accuracy using metrics like R-squared, RMSE, and MAE.
   - Validate the model through cross-validation to ensure robustness.

5. **Deployment**:
   - Deploy the model using a web-based interface where users can input house features to get price predictions.
   - Design a dashboard for visualizing house market trends and predictive analysis.

### Key Components
1. **Exploratory Data Analysis (EDA)**:
   - Scatter plots, correlation matrices, and box plots to reveal relationships between variables.

2. **Machine Learning Modeling**:
   - Linear regression, Random Forest, and Gradient Boosting models are considered.
   - Hyperparameter tuning for optimal model performance.

3. **Web-Based Dashboard**:
   - Interactive visualizations for users to explore house prices and model predictions.
   - Inputs for predicting house prices based on user-provided features.

## User Stories

To ensure the project aligns with the business requirements, several user stories have been identified. These stories help to bridge the business needs with the predictive analytics and visualizations in the project.

1. **As a homeowner in Ames, Iowa, I want to understand which house features are most important in determining sale prices so that I can prioritize renovations to increase the value of my property.**

2. **As a potential buyer, I want to predict the sale price of a house based on its features, so that I can budget effectively and make informed purchasing decisions.**

3. **As a real estate agent, I want to identify trends in house pricing across different neighborhoods in Ames, Iowa, so that I can provide accurate market insights to my clients.**

4. **As a data analyst, I want to build and validate a predictive model that estimates house prices accurately, so I can offer data-driven recommendations to stakeholders.**

5. **As a financial advisor, I want to understand the impact of house conditions and features on market value, so that I can provide better investment advice to clients looking to buy or sell real estate.**

These user stories outline the various stakeholders involved and their objectives, ensuring that the project caters to different perspectives and needs.

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|
## reasons behind dataset
The dataset contains various features that describe the characteristics of homes in Ames, Iowa, such as the total square footage, overall quality rating, garage area, year built, and neighborhood. The goal is to predict the sale price of a house based on these features, addressing the business problem of accurately valuing properties. This aligns with the CRISP-DM framework's "Business Understanding" phase, where the objective is to gain insight into property valuation trends to support homeowners, buyers, and real estate agents in making informed decisions.

The dataset features include:

- Total Square Footage (TotalSF): Indicates the overall living space in the house.
- Overall Quality (OverallQual): Measures the material and finish of the house, rated on a scale.
- Garage Area (GarageArea): Reflects the size of the garage.
- Year Built (YearBuilt): The construction year, representing the house's age.
- Neighborhood: The geographical location, influencing the property's market value.
- The business requirement is to provide an accurate and data-driven valuation tool for real estate transactions in Ames.

# Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypothesis and how to validate?

There is more than 1 hypothesis to consider with this. See below for my hypothesis and the validations
### Hypothesis 1: Larger Houses Have Higher Sale Prices

**Reasoning:** Generally, larger homes tend to sell for more because they offer more living space, which is a key factor in determining property value.
**Validation:** This can be validated by analyzing the correlation between the total square footage (including above-ground living area and basement) and the sale price. A positive correlation would support this hypothesis. We will create scatter plots and compute correlation coefficients to visualize and quantify the relationship.

### Hypothesis 2: Location Significantly Affects House Prices

**Reasoning:** The location of a house often has a significant impact on its price, with certain neighborhoods or proximity to amenities being more desirable and, therefore, more expensive.
**Validation:** This hypothesis can be validated by examining the correlation between the sale price and categorical variables representing location, such as the neighborhood or proximity to schools and parks. We can use box plots to compare sale prices across different neighborhoods and calculate ANOVA or t-tests to determine if the differences are statistically significant.
### Hypothesis 3: Houses with Higher Quality and Condition Ratings Have Higher Sale Prices

**Reasoning:** Homes that are well-maintained and built with high-quality materials are likely to command higher prices.
**Validation:** This can be validated by analyzing the correlation between sale price and variables related to quality and condition, such as overall quality (OverallQual), exterior quality (ExterQual), and overall condition (OverallCond). We will use scatter plots and correlation coefficients to measure the strength of these relationships.

### Hypothesis 4: Newer Houses Are Priced Higher Than Older Houses

**Reasoning:** Newer homes often have modern amenities, better insulation, and require less immediate maintenance, making them more attractive to buyers.
**Validation:** This hypothesis can be validated by examining the correlation between the year built (YearBuilt) and the sale price. We will analyze the relationship using scatter plots and correlation coefficients to see if newer homes consistently sell for more.

### Hypothesis 5: Houses with Additional Features (e.g., Pools, Garages, Fireplaces) Have Higher Sale Prices

**Reasoning:** Additional features like swimming pools, garages, and fireplaces add value to a home and are often reflected in a higher sale price.
**Validation:** This hypothesis can be validated by analyzing the correlation between sale price and the presence of additional features (e.g., PoolArea, GarageArea, FireplaceQu). We will use box plots and correlation coefficients to assess the impact of these features on the sale price.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

Business Requirement 1: Understanding How House Attributes Correlate with Sale Price
Objective: The client wants to identify which attributes of houses are most strongly correlated with their sale prices to understand what factors drive property values in Ames, Iowa.

### Potential Data Visualizations:

* Correlation Heatmap: A heatmap displaying the correlation coefficients between various house attributes (e.g., size, location, quality) and the sale price. This provides a quick overview of which factors are most strongly correlated with house prices.

* Scatter Plots: Individual scatter plots for key continuous variables (e.g., total square footage, year built) against sale price. This will help in visualizing the linear relationships and any potential outliers.

* Box Plots: Box plots for categorical variables (e.g., neighborhood, house style) against sale price. This will allow the client to see how sale prices vary across different categories and identify which categories are associated with higher or lower prices.
Histograms/Bar Charts: Histograms for continuous variables (e.g., living area, lot size) and bar charts for categorical variables (e.g., exterior quality, overall condition) to show the distribution of these attributes in the dataset.

Machine Learning Tasks:

Feature Importance Analysis: Using a machine learning model like Random Forest or Gradient Boosting to assess the importance of different features in predicting sale price. This will quantitatively show which attributes have the most impact on the price.
Regression Analysis: Multiple linear regression to quantify the relationships between house attributes and sale price, providing insights into how much each feature contributes to the price.
Rationale:

The visualizations help the client intuitively understand the relationships between house attributes and sale price, while the machine learning tasks provide a more rigorous and quantitative analysis of these relationships.

Business Requirement 2: Predicting House Sale Prices
Objective: The client wants to accurately predict the sale prices of the four inherited houses and any other house in Ames, Iowa, to ensure they are priced competitively and in line with the market.

### Potential Data Visualizations:

* Prediction Error Plot: A plot comparing predicted sale prices versus actual sale prices for the test dataset. This will help visualize how well the model is performing and where it might be over- or under-predicting.

* Residual Plots: Residual plots to analyze the difference between predicted and actual prices. This helps in understanding the distribution of errors and identifying any systematic biases in the predictions.

### Machine Learning Tasks:

Model Training: Training various regression models (e.g., Linear Regression, Ridge Regression, Random Forest, Gradient Boosting) to predict sale prices based on the identified significant features.

Model Validation: Evaluating the performance of the trained models using metrics like R-squared, RMSE, and MAE on a test dataset. This ensures the model generalizes well to unseen data.
Hyperparameter Tuning: Fine-tuning the model parameters to optimize performance, ensuring the most accurate predictions possible.

Final Model Deployment: Once the best model is selected, it will be used to predict the sale prices of the four inherited houses.

### Rationale:

* Accurate predictions are crucial for the client to set the right price for the inherited properties. The visualizations offer a clear view of the model's performance, while the machine learning tasks focus on building and validating a model that can reliably predict sale prices.
Summary

* Data Visualizations provide intuitive, easy-to-understand insights into the relationships between house attributes and sale prices, helping to identify key factors that affect pricing.
Machine Learning Tasks focus on building, validating, and fine-tuning predictive models that can accurately estimate house prices based on the identified features, ensuring that the client can make informed pricing decisions for the inherited properties.

## Machine Learning Business Case

### Business Objective
The client aims to maximize the sale prices of four inherited properties in Ames, Iowa. Since she is unfamiliar with the local market, a data-driven approach will help understand the key factors influencing house prices and enable accurate pricing for these properties. This will minimize the risk of financial loss due to overpricing or underpricing.

### Problem Statement
Accurate prediction of house sale prices is crucial for pricing the properties competitively. Without understanding the Ames real estate market, there is a risk of setting inappropriate prices, potentially leading to prolonged time on the market (if overpriced) or lost revenue (if underpriced). The predictive model will help the client achieve market-aligned pricing.

### ML Solution Overview
The solution involves building a machine learning model using historical real estate data from Ames. By incorporating features such as square footage, quality, garage area, year built, and neighborhood, the model will learn the relationships that influence house prices. The model will then predict the prices for the inherited properties based on these attributes.

### Learning Method
Several regression techniques will be used to identify the most effective model. Potential algorithms include:
- **Linear Regression** for its simplicity in providing a baseline.
- **Random Forest** for handling non-linear relationships.
- **Gradient Boosting** for improving predictive accuracy through ensemble learning.

### Ideal Outcome
The model should accurately estimate sale prices, reducing the difference between predicted and actual sale prices, which will ensure competitive pricing.

### Success Metrics
Key performance indicators for evaluating the model include:
- **R-squared (R²)**: Should be high, indicating that the model explains a significant portion of the price variance.
- **Root Mean Squared Error (RMSE)**: Should be low, reflecting a small average prediction error.
- **Mean Absolute Error (MAE)**: Should also be minimized, indicating accurate predictions across the board.

### Model Output and Relevance
The model's output—predicted sale prices—will guide the client in setting listing prices. These predictions help understand how various factors, such as location and house quality, impact pricing, offering valuable market insights.

### Heuristics and Training Data
Historical data from the Ames housing market, containing features like "Total Square Footage" and "Neighborhood," will be used. Important steps include:
- Data cleaning to handle missing values.
- Feature selection based on correlation analysis.
- Cross-validation to prevent overfitting.

### Risks and Considerations
- **Data Quality**: Incomplete or erroneous data may degrade model accuracy. Preprocessing will be necessary.
- **Overfitting**: Regularization and cross-validation techniques will help avoid fitting the model too closely to the training data.
- **Market Changes**: The model's predictions are based on past trends, which may not fully reflect future shifts. Periodic updates to the model may be required.

### Implementation Plan
1. **Data Collection and Preprocessing**: Clean historical data and engineer relevant features.
2. **Exploratory Data Analysis (EDA)**: Investigate data distributions, correlations, and outliers.
3. **Model Selection and Training**: Train multiple models, selecting the best one based on validation performance.
4. **Model Validation and Hyperparameter Tuning**: Optimize the selected model using techniques like grid search.
5. **Prediction and Reporting**: Use the trained model to predict sale prices, providing a report with insights.
6. **Monitoring and Maintenance**: Regularly track the model's performance and update it as new data becomes available.

### Cost-Benefit Analysis
- **Costs**: Include time for data collection, model development, and deployment.
- **Benefits**: Consist of increased revenue from accurate pricing, lower risk of financial loss, and better market insights.

### Conclusion
The machine learning model will enable the client to price the inherited properties competitively, leveraging data insights to maximize potential returns while reducing the risk of pricing errors. This solution will provide confidence and a strategic advantage in navigating the Ames real estate market.


## Dashboard Design

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.
## Potential Problems
As i may have to connect to the Code institute custom database, there may be an issue with seeing the application incase the CI database has any issues or potential problems which could cause me to not be able to get all of the application built.
## Fixed Bugs
The main bugs i have fixed are making sure that each cell of code does not show any errors and outputs something relating to housing.
## Testing
This section is all about the tests i have performed to the application manually and via different browsers.
### Manual Testing
I have not been able to fully test the application in terms of data loading and correct outputs, but i have been able to test if it loads on different browsers and that their is no visible errors being shown.
### Browser Testing
Heritige Housing has been extensively tested on different browsers including edge, chrome, safari and firefox.
| Browser | Layout | Functionality |
|:.......:|:......:|:.............:|
|  Chrome |   ✔    |        ✔     |
| Safari  |   ✔    |        ✔     |
| Edge    |   ✔    |        ✔     |
| FireFox |   ✔    |        ✔     |
| Ecosia  |   ✔    |        ✔     |
## Deployment
This section talks about deploying to heroku
### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* **pandas** - Used for data manipulation and analysis, especially for handling tabular data in DataFrames.
* **os** - Provides functions to interact with the operating system, such as file and directory operations.
* **streamlit** - A web framework for building interactive data applications and visualizations.
* **numpy** - Supports numerical computations, providing functions for working with arrays and mathematical operations.
* **seaborn** - A visualization library built on top of Matplotlib, offering enhanced plotting capabilities, especially for statistical graphics.
* **matplotlib.pyplot** - A foundational library for creating static, interactive, and animated visualizations in Python.
* **plotly.express** - Allows for the creation of interactive plots and dashboards, making it suitable for dynamic data exploration.
* **sklearn.model_selection** - Contains functions for splitting data into training and testing sets, as well as for performing cross-validation.
* **sklearn.ensemble** - Provides ensemble learning algorithms like Random Forest and Gradient Boosting for regression tasks.
* **sklearn.linear_model** - Implements linear models, including Linear Regression and Ridge Regression, for predictive modeling.
* **sklearn.metrics** - Offers functions to evaluate the performance of machine learning models using metrics such as R-squared, RMSE, and MAE.
* **sklearn.preprocessing** - Includes tools for data preprocessing, such as scaling and encoding, to prepare data for machine learning models.

## Credits

* In this section i am referencing where all of my content and media came from to avoid plagerism and having to resubmit this project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

## Contributing

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements
- My mentor precious Ijege
- My amazing peers who helped me when times were tough
- The 2 project walkthroughs to give me insight as to how to deploy the application
- Code institutes assessment criteria (to help me double check if i have done everything for pass/merit/distinction)
- The Amazing Beth cottell who helped me try to figure out a fix to an error when it came to deploying the application.
- Roman from the Code Institute Tutor Support team who helped fix the issue of deploying.