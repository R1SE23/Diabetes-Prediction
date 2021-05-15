from plot import *
from model_pipeline import *
from plot_performance import *
import webbrowser

colab = 'https://colab.research.google.com/drive/1bKjkpyKinBOElD19BQ4njyyTp1vj0f6W?usp=sharing'
pycaret = 'https://colab.research.google.com/drive/1qwfNPrJOyB6NHYmt6gIYSNyQamXiphi5?usp=sharing'


def main():
    # create a title and sub-title
    st.write("""
    # Diabetes Detection
    Detect if someone has diabetes using machine learning and python!
    """)
    my_page = st.sidebar.radio('Page Navigation', ['Model Prediction', 'Data Analytics', 'Modeling with Pycaret'])

    if my_page == 'Model Prediction':
        # get the data
        data = pd.read_csv('diabetes.csv')
        # set a subheader
        st.subheader('Data Information:')
        # show the data as a table
        st.dataframe(data)
        # show statistics on the data
        st.write(data.describe())

        # Pipeline for modeling
        D = data[(data['Outcome'] != 0)]
        H = data[(data['Outcome'] == 0)]
        data = replace_missing(data)
        data = replace_median(data)
        data = feature_engineering(data)
        data = prepare_data(data)
        # Def X and Y
        X = data.drop('Outcome', 1)
        y = data['Outcome']

        # best params - > https://colab.research.google.com/drive/1bKjkpyKinBOElD19BQ4njyyTp1vj0f6W?usp=sharing
        params = {'colsample_bytree': 0.5062062905660482,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_samples': 100,
        'min_child_weight': 1,
        'n_estimators': 1500,
        'num_leaves': 17,
        'reg_alpha': 0.1,
        'reg_lambda': 0,
        'subsample': 0.6027338166838856}

        # base model
        model = lgbm.LGBMClassifier(**params)
        model.fit(X, y.values.ravel())

        st.subheader('Model Performance')
        # select model to see its performance
        selectmodel = st.selectbox('Select Model', ['Fine-tuned LGBM'])
        if selectmodel == 'Fine-tuned LGBM':
            # set a subheader
            st.write(f'Model Performance: {selectmodel}')
            # model performance
            mdp = model_performance(model, 'LightGBM', X, y)
            st.plotly_chart(mdp)
            scr = scores_table(model, 'LightGBM', X, y)
            st.plotly_chart(scr)
    
        # user input data
        def get_user_input():
            pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
            glucose = st.sidebar.slider('glucose', 0.0, 199.0, 117.0)
            blood_pressure = st.sidebar.slider('blood_pressure', 0.0, 122.0, 72.0)
            skin_thickness = st.sidebar.slider('skin_thickness', 0.0, 99.0, 23.0)
            insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
            BMI = st.sidebar.slider('BMI', 0.0, 67.0, 32.0)
            DPF = st.sidebar.slider('DPF', 0.078, 2.420, 0.372)
            age = st.sidebar.slider('age', 21, 81, 29)
            # Store a dictionary into a variable
            user_data = {'Pregnancies': pregnancies,
                        'Glucose': glucose,
                        'BloodPressure': blood_pressure,
                        'SkinThickness': skin_thickness,
                        'Insulin': insulin,
                        'BMI': BMI,
                        'DiabetesPedigreeFunction': DPF,
                        'Age': age}

            # Transform the data into a dataframe
            features = pd.DataFrame(user_data, index = [0])
            return features

        # Store the user input into a variable
        form = st.sidebar.form(key='my-form')
        submit = form.form_submit_button('Submit value for prediction')
        user_input = get_user_input()
        # set a subheader and display the users input
        st.subheader('User Input:')
        st.write(user_input)

        if submit:
            ui = feature_engineering(user_input)
            ui = prepare_data_ui(ui)
            prediction = model.predict(ui)
            # Set a subheader and display the classification
            st.subheader('Prediction Result ')
            if prediction == 0:
                st.write('This person is healthy')
            else:
                st.write('This person has high risk of diabetes')


    elif my_page == 'Data Analytics':
        df = pd.read_csv('diabetes.csv')
        st.title('Data Analytics')
        # if st.button('Click here to view the full Data Analysis of Diabetes Dataset'):
        #     webbrowser.open_new_tab(colab)
        selectbox = st.selectbox('Select', ['Distribution','Feature Distribution','Feature Correlation'])
        if selectbox == 'Distribution':
            tc = target_count(df)
            st.plotly_chart(tc)
            tp = target_percent(df)    
            st.plotly_chart(tp)
        elif selectbox == 'Feature Distribution':
            plot_all_feature(df)
        elif selectbox == 'Feature Correlation':
            image = Image.open('img/corr.png')
            st.image(image, caption='feature-correlation', use_column_width=True)
            

    elif my_page == "Modeling with Pycaret":
        # set a subheader and display the users input
        st.subheader('Model Perfomance: Before applying Feature Engineering')
        # display an image
        image = Image.open('pycaret-screenshots/base-model-acc.JPG')
        st.image(image, caption='model',use_column_width=True)
        # set a subheader and display the users input
        st.subheader('Model Perfomance: After applied Feature Engineering')
        # display an image
        image = Image.open('pycaret-screenshots/after-feature-engineering-Pycaret.JPG')
        st.image(image, caption='model',use_column_width=True)

        if st.button('Click here to view the Colab code for the Modeling part with the help of Pycaret'):
            webbrowser.open_new_tab(pycaret)
        


if __name__ == "__main__":
    main()




