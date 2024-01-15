from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer

# Load the dataset
df = pd.read_csv('data_updated_split.csv')

# Convert categorical variables to numerical codes
brand_options = ['Select brand', 'vivo', 'POCO', 'APPLE', 'MOTOROLA', 'OnePlus', 'REDMI', 'Infix', 'realme', 'SAMSUNG',
                 'Nokia','I', 'itel', 'Jio', 'IQOO', 'Kechaoda', 'Nothing', 'Micromax', 'Google', 'Tecno', 'Cellecor',
                 'LAVA', 'OPPO', 'MTR', 'KARBONN', 'Snexian', 'BlackZone']
df['Brand'] = df['Brand'].astype('category').cat.codes
df['Color'] = df['Color'].astype('category').cat.codes

# Separate features (X) and target variable (Y)
X = df.drop(columns='Price(rupees)')
Y = df['Price(rupees)']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Impute missing values with KNN
imputer = KNNImputer()
X_imputed_knn = imputer.fit_transform(X_scaled)

# Feature selection using RFE
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rfe = RFE(rf, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_imputed_knn, Y)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_rfe, Y, test_size=0.3, random_state=0)

# Create and fit the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, Y_train)

# R-squared for training set
r2_train = r2_score(Y_train, rf_model.predict(X_train))

# R-squared for testing set
r2_test = r2_score(Y_test, rf_model.predict(X_test))

# Predict the target variable on the test set
Y_pred_test = rf_model.predict(X_test)

# Evaluate the model on the test set
mae_test = mean_absolute_error(Y_test, Y_pred_test)
mse_test = mean_squared_error(Y_test, Y_pred_test)

# Streamlit App
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱")
st.title('Mobile Price Prediction App')

# Sidebar Layout
with st.sidebar:
    st.sidebar.header('Input Features')

    # Brand dropdown
    brand = st.selectbox('Brand', brand_options)

    # Generation input
    generation = st.number_input('Generation (G)', min_value=1, max_value=10, value=5)

    # Color input (case insensitive)
    color_placeholder = 'e.g., Black'
    color = st.text_input('Color', color_placeholder, key='color').title()

    # Ratings input
    ratings = st.slider('Ratings', min_value=0, max_value=5, step=1, value=4)

    # RAM input
    ram = st.slider('RAM (GB)', min_value=0, max_value=12, step=1, value=4)

    # ROM input
    rom_options = [8, 16, 32, 64, 128, 256, 512]
    rom = st.selectbox('ROM (GB)', rom_options, index=3)

    # Display input
    display = st.number_input('Display (inch)', min_value=1.0, max_value=10.0, format="%.1f", value=5.5)

    # Camera input
    camera = st.slider('Camera (MP)', min_value=1, max_value=108, step=1, value=12)

    # Battery input
    battery_options = [800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    battery = st.selectbox('Battery (mAh)', battery_options, index=4)

    # Reviews input
    reviews = st.slider('Reviews', min_value=0, max_value=100000, value=5000)

    # Discount input
    discount = st.slider('Discount (% off)', min_value=0, max_value=90, value=10)

# Prepare user input for prediction
user_input = pd.DataFrame({
    'Brand': [brand],
    'Generation(G)': [generation],
    'Color': [color],
    'Ratings': [ratings],
    'RAM(GB)': [ram],
    'ROM(GB)': [rom],
    'Display(inch)': [display],
    'Camera(MP)': [camera],
    'Battery(mAh)': [battery],
    'Reviews': [reviews],
    'Discount(%off)': [discount]
})

# Convert categorical variables to numerical codes
user_input['Brand'] = pd.Categorical(user_input['Brand'], categories=brand_options).codes
user_input['Color'] = user_input['Color'].astype('category').cat.codes

# Feature Scaling
user_input_scaled = scaler.transform(user_input)

# Impute missing values with KNN
user_input_imputed = imputer.transform(user_input_scaled)

# Feature selection using RFE
user_input_rfe = rfe.transform(user_input_imputed)

# Make Prediction only when user inputs values
if st.button('Predict Price') and brand != 'Select brand':
    # Make Prediction
    predicted_price = rf_model.predict(user_input_rfe)

    # Display Predicted Price
    st.success(f'Predicted Price: {predicted_price[0]:,.2f} rupees')

# Display user input
st.subheader('Selected Input Features:')
st.write(user_input)

# Additional Metrics
st.subheader('Model Evaluation Metrics')
st.text(f"R-squared on Training Set: {r2_train:.4f}")
st.text(f"R-squared on Test Set: {r2_test:.4f}")
st.text(f"Mean Absolute Error (MAE) on Test Set: {mae_test:.2f}")
st.text(f"Mean Squared Error (MSE) on Test Set: {mse_test:.2f}")

# Dashboard Section
st.subheader('Various Statistics')
st.text(f"Mean Price: {Y.mean():,.2f} rupees")
st.text(f"Minimum Price: {Y.min():,.2f} rupees")
st.text(f"Maximum Price: {Y.max():,.2f} rupees")
st.text(f"Price Standard Deviation: {Y.std():,.2f} rupees")

# Distribution of Prices
st.subheader('Price Distribution')
plt.figure(figsize=(8, 6))
plt.hist(Y, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Prices')
plt.xlabel('Price (rupees)')
plt.ylabel('Frequency')
st.pyplot(plt)

# Feature Importance
st.subheader('Feature Importance')
feature_importance = pd.Series(rfe.estimator_.feature_importances_, index=X.columns[rfe.support_])
feature_importance = feature_importance.sort_values(ascending=False)
st.bar_chart(feature_importance)
