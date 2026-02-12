import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
df=pd.read_csv("insurance.csv")
target=df['charges']
features=df.drop(['charges'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.3,random_state=42)

cat=['sex','smoker','region']
num=['age','bmi','children']

num_pipeline=Pipeline([

    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
]
)
cat_pipeline=Pipeline([
    ('cat',OneHotEncoder(drop='first'))
])
preprocessing=ColumnTransformer(
    transformers=[
        ('num',num_pipeline,num),
        ('cat',cat_pipeline,cat)
    ]
)
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('classifier',RandomForestRegressor(n_estimators=1000,random_state=42))
])

pipeline.fit(x_train,y_train)
y_pred = pipeline.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R² Score:", r2*100)


result=pd.DataFrame(
    {
    "Actual":y_test.values,
    "predict":y_pred
    }
)
result.to_csv("output.csv",index=False)
print("Successfully Save the File...")
joblib.dump(pipeline,'insurance.pkl')