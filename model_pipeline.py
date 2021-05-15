from plot_performance import *

def replace_missing(data):
    data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
    return data

def median_target(data,var):   
    temp = data[data[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

def replace_median(data):
    null_columns = ['BloodPressure', 'BMI', 'SkinThickness', 'Glucose', 'Insulin']
    for i in null_columns:
        f = median_target(data, i)
        data.loc[(data['Outcome'] == 0 ) & (data[i].isnull()), i] = f[[i]].values[0][0]
        data.loc[(data['Outcome'] == 1 ) & (data[i].isnull()), i] = f[[i]].values[1][0]
    return data

def feature_engineering(data):
    data.loc[:,'N1']=0
    data.loc[(data['Age']<=30) & (data['Glucose']<=120),'N1']=1

    data.loc[:,'N2']=0
    data.loc[(data['BMI']<=30),'N2']=1

    data.loc[:,'N3']=0
    data.loc[(data['Age']<=30) & (data['Pregnancies']<=6),'N3']=1

    data.loc[:,'N3_1']=0
    data.loc[(data['Glucose']<=110) & (data['Pregnancies']<=5),'N3_1']=1

    data.loc[:,'N4']=0
    data.loc[(data['Glucose']<=105) & (data['BloodPressure']<=80),'N4']=1

    data.loc[:,'N4_1']=0
    data.loc[(data['Age']<=30) & (data['Pregnancies']<=6),'N4_1']=1

    data.loc[:,'N5']=0
    data.loc[(data['SkinThickness']<=20) ,'N5']=1

    data.loc[:,'N6']=0
    data.loc[(data['BMI']<30) & (data['SkinThickness']<=20),'N6']=1

    data.loc[:,'N7']=0
    data.loc[(data['Glucose']<=105) & (data['BMI']<=30),'N7']=1

    data.loc[:,'N7_1']=0
    data.loc[(data['BMI']<30) & (data['SkinThickness']<=20),'N7_1']=1

    data.loc[:,'N9']=0
    data.loc[(data['Insulin']<200),'N9']=1

    data.loc[:,'N10']=0
    data.loc[(data['BloodPressure']<80),'N10']=1

    data.loc[:,'N11']=0
    data.loc[(data['Pregnancies']<4) & (data['Pregnancies']!=0) ,'N11']=1

    # highly correlate data

    data['N0'] = data['BMI'] * data['SkinThickness']

    data['N8'] =  data['Pregnancies'] / data['Age']

    data['N13'] = data['Glucose'] / data['DiabetesPedigreeFunction']

    data['N12'] = data['Age'] * data['DiabetesPedigreeFunction']

    data['N14'] = data['Age'] / data['Insulin']

    data['N15'] = data['BMI'] / data['Insulin']
    return data

@st.cache 
def prepare_data(data):
    cat_cols   = data.nunique()[data.nunique() < 12].keys().tolist()
    cat_cols   = [x for x in cat_cols]

    #numerical columns
    num_cols   = [x for x in data.columns if x not in cat_cols]

    #Scaling Numerical columns
    std = StandardScaler()
    scaled = std.fit_transform(data[num_cols])
    scaled = pd.DataFrame(scaled,columns=num_cols)

    #dropping original values merging scaled values for numerical columns
    df_data_og = data.copy()
    data = data.drop(columns = num_cols,axis = 1)
    data = data.merge(scaled,left_index=True,right_index=True,how = "left")
    return data

# data preparation for user
def prepare_data_ui(data):
    cat = ['N1', 'N2', 'N3', 'N3_1', 'N4', 'N4_1', 'N5', 'N6', 'N7', 'N7_1', 'N9', 'N10', 'N11']
    num = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'N0', 'N8', 'N13', 'N12', 'N14', 'N15']
    std = StandardScaler()
    scaled = std.fit_transform(data[num].values[0][:, np.newaxis])
    scaled = pd.DataFrame(scaled.T, columns=num)
    df_data_og = data.copy()
    data = data.drop(columns = num,axis = 1)
    data = data.merge(scaled, left_index=True, right_index=True, how = "left")
    return data
  