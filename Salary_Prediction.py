####################################  SALARY PREDICTION on HITTERS DATASET  #######################################

# Imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve,cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder
import warnings
from sklearn.impute import KNNImputer
import missingno as msno
warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings(action='ignore', category=Warning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.WarningMessage
from warnings import  filterwarnings
filterwarnings("ignore")

df = pd.read_csv("datasets/hitters.csv")
df.head()

# 1) EDA
# Veri setine genel bir göz atalım
check_df(df)

# Betimsel istatistiklere detaylı bir göz atalım.
df.describe([0.01, 0.05,0.25, 0.75, 0.90, 0.99]).T

# Kategorik ve Numerik Kolonları Yakalama
cat_cols, num_cols, cat_but_car = grab_cols(df)

# Aykırı Değerler - Outliers
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    if df[df[col] > up_limit].any(axis=None):
        print(col,"YES",(low_limit,up_limit))
    elif df[df[col] < low_limit].any(axis=None):
        print(col,"YES",(low_limit,up_limit))
    else:
        print(col,"NO",(low_limit,up_limit))

# Görsel olarak eksik değerleri gözlemleyelim
msno.matrix(df)
msno.bar(df)
plt.show()

# Değişkenlerin Birbirleriyle olan korelasyona göz atalım.
corr = df.corr()
sns.heatmap(corr)
plt.show()

# Feature Engineering

def hitters_script(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    for i in num_cols:
        df[i] = df[i].add(1)
    df.columns = [col.upper() for col in df.columns]

    # RATIO OF VARIABLES

    # CAREER RUNS RATIO
    df["NEW_C_RUNS_RATIO"] = df["RUNS"] / df["CRUNS"]
    # CAREER BAT RATİO
    df["NEW_C_ATBAT_RATIO"] = df["ATBAT"] / df["CATBAT"]
    # CAREER HİTS RATİO
    df["NEW_C_HITS_RATIO"] = df["HITS"] / df["CHITS"]
    # CAREER HMRUN RATİO
    df["NEW_C_HMRUN_RATIO"] = df["HMRUN"] / df["CHMRUN"]
    # CAREER RBI RATİO
    df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]
    # CAREER WALKS RATİO
    df["NEW_C_WALKS_RATIO"] = df["WALKS"] / df["CWALKS"]

    # CAREER RATİO
    df["NEW_C_HITPERATBAT"] = df["CHITS"] / df["CATBAT"]
    df["NEW_C_RBIPERHITS"] = df["CRBI"] / df["CHITS"]
    df["NEW_C_RUNSPERHITS"] = df["CRUNS"] / df["CHITS"]
    df["NEW_C_HMRUNPERHITS"] = df["CHMRUN"] / df["CHITS"]
    df["NEW_C_ATBATPERHMRUN"] = df["CATBAT"] / df["CHMRUN"]

    # Annual Averages-
    df["NEW_CATBAT_MEAN"] = df["CATBAT"] / df["YEARS"]
    df["NEW_CHITS_MEAN"] = df["CHITS"] / df["YEARS"]
    df["NEW_CHMRUN_MEAN"] = df["CHMRUN"] / df["YEARS"]
    df["NEW_CRUNS_MEAN"] = df["CRUNS"] / df["YEARS"]
    df["NEW_CRBI_MEAN"] = df["CRBI"] / df["YEARS"]
    df["NEW_CWALKS_MEAN"] = df["CWALKS"] / df["YEARS"]

    # SEASON RATİO
    df["NEW_HITPERATBAT"] = df["HITS"] / df["ATBAT"]
    df["NEW_RBIPERHITS"] = df["RBI"] / df["HITS"]
    df["NEW_RUNSPERHITS"] = df["RUNS"] / df["HITS"]
    df["NEW_HMRUNPERHITS"] = df["HMRUN"] / df["HITS"]
    df["NEW_ATBATPERHMRUN"] = df["ATBAT"]+1 / df["HMRUN"]
    df["NEW_TOTAL_CHANCES"] = df["ERRORS"] + df["PUTOUTS"] + df["ASSISTS"]

    # PLAYER YEARS LEVEL
    df.loc[(df["YEARS"] <= 2),"NEW_YEARS_LEVEL"] = "Junior"
    df.loc[(df["YEARS"] >2 ) & (df['YEARS'] <= 5 ), "NEW_YEARS_LEVEL"] = "Mid"
    df.loc[(df["YEARS"] >5 ) & (df['YEARS'] <= 10 ), "NEW_YEARS_LEVEL"] = "Senior"
    df.loc[(df["YEARS"] >10) , "NEW_YEARS_LEVEL"] = "Expert"

    # PLAYER YEARS LEVEL X DIVISION

    df.loc[(df["NEW_YEARS_LEVEL"] == "Junior") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Junior-East"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Junior") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Junior-West"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Mid") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Mid-East"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Mid") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Mid-West"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Senior") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Senior-East"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Senior") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Senior-West"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Expert") & (df["DIVISION"] == "E"), 'NEW_DIV_CAT'] = "Expert-East"
    df.loc[(df["NEW_YEARS_LEVEL"] == "Expert") & (df["DIVISION"] == "W"), 'NEW_DIV_CAT'] = "Expert-West"

    # Player Promotion to Next League
    df.loc[(df["LEAGUE"] == "N" ) & (df["NEWLEAGUE"] == "N" ), "NEW_PLAYER_PROGRESS"] = "StandN"
    df.loc[(df["LEAGUE"] == "A" ) & (df["NEWLEAGUE"] == "A" ), "NEW_PLAYER_PROGRESS"] = "StandA"
    df.loc[(df["LEAGUE"] == "N" ) & (df["NEWLEAGUE"] == "A" ), "NEW_PLAYER_PROGRESS"] = "Descend"
    df.loc[(df["LEAGUE"] == "A" ) & (df["NEWLEAGUE"] == "N" ), "NEW_PLAYER_PROGRESS"] = "Ascend"

    return df

df = hitters_script(df)

# Yeni Değişkenler Oluşturduğumuz için Kategorik ve Numerik Kolonları Tekrar Yakalıyoruz.
cat_cols, num_cols, cat_but_car = grab_cols(df)

# Aykırı değerleri yeni değişkenler oluşturduktan sonra eşik değerlerine baskılamayı terchi ediyoruz.
# Replace with Thresholds
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    df.loc[(df[col] < low_limit), col] = low_limit
    df.loc[(df[col] > up_limit), col] = up_limit
    if df[df[col] > up_limit].any(axis=None):
        print(col,"YES",(low_limit,up_limit))
    elif df[df[col] < low_limit].any(axis=None):
        print(col,"YES",(low_limit,up_limit))
    else:
        print(col,"NO",(low_limit,up_limit))

### Encoding

# Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)

# Rare Encoding
df = rare_encoder(df, 0.01,cat_cols)

# One - hot Encodşng
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

# Yeni Değişkenler Oluşturduğumuz için Kategorik ve Numerik Kolonları Tekrar Yakalıyoruz.
cat_cols, num_cols, cat_but_car = grab_cols(df)

# KNN
imputer = KNNImputer(n_neighbors=5)
df_filled = imputer.fit_transform(df)
df = pd.DataFrame(df_filled, columns=df.columns)

# Local Outlier
lof = LocalOutlierFactor(n_neighbors=20)
lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:40]
threshold = np.sort(df_scores)[7]
threshold
outlier = df_scores > threshold
df = df[outlier]

# Eksik Değerleri gözlemleyelim
df.isnull().sum()

# Veri setini Scale ediyoruz
# Robust Scaler
num_cols.remove("SALARY")
for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])

# Model

y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)


###### Base Models
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')), # objective amaç fonksiyonu yani coss - loss fonksiyonu
          ("LightGBM", LGBMRegressor()),
          # ("CatBoost", CatBoostRegressor(verbose=False))
          ]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# Random Customizer
lgbm_model = LGBMRegressor()

# geniş bir parametre aralığı veriyoruz.
lgbm_random_params = {"max_depth": range(-10, 10),
                   "learning_rate": [0.1,0.2,0.3,0.01,0.0001,1],
                   "n_estimators": [100,200,400,500,600,1000],
                     "subsample": [-2,-1,0,1,2,3,4],
                    "boosting_type" : ["gbdt","goss","dart"]}


# Random searchcv olası kombinasyonlardan istediğimiz kadar seçmemizi sağlar biz 20 tane yazmışız
lgbm_random = RandomizedSearchCV(estimator=lgbm_model,
                               param_distributions=lgbm_random_params,
                               n_iter=50,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)
lgbm_random.fit(X, y)
# En iyi hiperparametre değerleri:
lgbm_random.best_params_


# En iyi skor
lgbm_random.best_score_

# Final Models
cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, None],
             "max_features": [5,  "auto"],
             "min_samples_split": [2, 5],
             "n_estimators": [100,600,700],
            "min_weight_fraction_leaf" : [0.1,0.0],
             "min_samples_leaf" : [3,1]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 0.8]}

lightgbm_params = {"max_depth": [1,3,5],
                   "learning_rate": [0.1,0.3,0.01],
                   "n_estimators": [100,200,500],
                    "subsample": [1,3,8],
                    "boosting_type" : ["gbdt","dart"]
                   }

regressors = [
              ("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(bootstrap=False), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(boosting_type="dart"), lightgbm_params)
              ]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model