from django.shortcuts import render,redirect
from Mainapp.models import*
from Userapp.models import*
from Adminapp.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
import os
import shutil
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from django.contrib import messages
from keras.models import load_model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .models import SVM,ANN
import joblib  # Assuming 'RF','DT','XGB is your model in models.py to store the metrics

#gradient boost machine algo for getting acc ,precession , recall , f1 score
# Create your views here.
def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('index')

def usergraph(req):
    return render(req,'admin/user-sentiment-graph.html')

def admindashboard(req):
    all_users_count =  User.objects.all().count()
    pending_users_count = User.objects.filter(User_Status = 'Pending').count()
    rejected_users_count = User.objects.filter(User_Status = 'removed').count()
    accepted_users_count =User.objects.filter(User_Status = 'accepted').count()
    Feedbacks_users_count= Feedback.objects.all().count()
    user_uploaded_images =Dataset.objects.all().count()
    return render(req,'admin/admin-dashboard.html',{'a' : all_users_count, 'b' : pending_users_count, 'c' : rejected_users_count, 'd' : accepted_users_count, 'e':Feedbacks_users_count, 'f':user_uploaded_images})

def pendingusers(req):
    pending = User.objects.filter(User_Status = 'Pending')
    paginator = Paginator(pending, 5) 
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req,'admin/admin-pending-users.html', { 'user' : post})

def delete_user(req, id):
    User.objects.get(User_id = id).delete()
    messages.warning(req, 'User was Deleted..!')
    return redirect('manageusers')

def accept_user(req, id):
    status_update = User.objects.get(User_id = id)
    status_update.User_Status = 'accepted'
    status_update.save()
    messages.success(req, 'User was accepted..!')
    return redirect('pendingusers')

def manageusers(req):
    manage_users  = User.objects.all()
    paginator = Paginator(manage_users, 5)
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req, 'admin/admin-manage-users.html', {"allu" : manage_users, 'user' : post})

def reject_user(req, id):
    status_update2 = User.objects.get(User_id = id)
    status_update2.User_Status = 'removed'
    status_update2.save()
    messages.warning(req, 'User was Rejected..!')
    return redirect('pendingusers')

def admin_datasetupload(req):
    return render(req,'admin/admin-upload-dataset.html')
def admin_dataset_btn(req):
    messages.success(req, 'Dataset uploaded successfully..!')
    return redirect('admin_datasetupload')

def adminfeedback(req):
    feed =Feedback.objects.all()
    return render(req,'admin/user-feedback.html', {'back':feed})

def adminsentiment(req):
    fee = Feedback.objects.all()
    return render(req,'admin/user-sentiment.html' , {'cat':fee})


def SVM_alg(req):
  return render(req,'admin/SVM_alg.html')

def ANN_alg(req):
  return render(req,'admin/ANN_alg.html')





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import joblib



def SVM_btn(req):
    # Load the larger synthetic dataset
    data = pd.read_csv('C:/Users/Dell/OneDrive/Desktop/DGSR/DGSR/DGSR/dataset/synthetic_solar_radiation_dataset_10000.csv')

    # Define independent (X) and dependent (y) variables
    X = data.drop(columns=['Global_Solar_Radiation_Wh/m²'])  # Features
    y = data['Global_Solar_Radiation_Wh/m²']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data for improved model performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ### 1. Support Vector Machine (SVM) Model

    # Initialize and train the SVM model
    svm_model = SVR(kernel='rbf')  # Using RBF kernel for non-linear regression
    svm_model.fit(X_train, y_train)

    # Make predictions with SVM
    y_pred_svm = svm_model.predict(X_test)

    # Calculate SVM performance metrics
    rmse_svm = np.sqrt(mean_squared_error(y_test, y_pred_svm))
    nrmse_svm = rmse_svm / (y_test.max() - y_test.min())  # NRMSE as a percentage
    mape_svm = mean_absolute_percentage_error(y_test, y_pred_svm)
    mbe_svm = np.mean(y_pred_svm - y_test)  # Mean Bias Error
    r2_svm = r2_score(y_test, y_pred_svm)

    # Prepare metrics and predictions for the template
    results = {
        "rmse_svm": round(rmse_svm, 2),
        "nrmse_svm": round(nrmse_svm * 100, 2),
        "mape_svm": round(mape_svm * 100, 2),
        "mbe_svm": round(mbe_svm, 2),
        "r2_svm": round(r2_svm * 100, 2),
    }

    print("\nSVM Model Performance:")
    print(f"RMSE: {rmse_svm:.2f}")
    print(f"NRMSE: {nrmse_svm * 100:.2f}%")
    print(f"MAPE: {mape_svm * 100:.2f}%")
    print(f"MBE: {mbe_svm:.2f}")
    print(f"R² (Correlation Coefficient): {r2_svm* 100:.2f}")

    # Save metrics in the database
    SVM.objects.create(
        name="SVM Antenna Structure Prediction",
        rmse=rmse_svm,
        nrmse=nrmse_svm,
        mape=mape_svm,
        mbe=mbe_svm,
       r2_coefficient=r2_svm,
    )

    # Fetch the latest database entry to pass to the template
    data = SVM.objects.last()

     # Save the trained model (optional)
    joblib.dump(svm_model, 'SVM.pkl')

    # Render the template with both results and saved database data
    return render(req, 'admin/SVM_alg.html', results )

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

def ANN_btn(req):
    # Load the trained ANN model
    ann_model = tf.keras.models.load_model('C:/Users/Dell/OneDrive/Desktop/DGSR/DGSR/DGSR/ANN_model2.h5')

    # Load the scaler
    scaler = joblib.load('C:/Users/Dell/OneDrive/Desktop/DGSR/DGSR/DGSR/scaler.pkl')

    # Load the dataset
    data = pd.read_csv('C:/Users/Dell/OneDrive/Desktop/DGSR/DGSR/DGSR/dataset/synthetic_solar_radiation_dataset_10000.csv')  

    # Prepare the dataset
    X = data.drop(columns=['Global_Solar_Radiation_Wh/m²'])
    y = data['Global_Solar_Radiation_Wh/m²']

    # Transform features using the saved scaler
    X_test = scaler.transform(X)  

    # Use the ANN model to predict
    y_pred_ann = ann_model.predict(X_test).flatten()
    

    # Compute performance metrics
    rmse_ann = np.sqrt(mean_squared_error(y, y_pred_ann))
    nrmse_ann = rmse_ann / (y.max() - y.min())
    mape_ann = mean_absolute_percentage_error(y, y_pred_ann)
    mbe_ann = np.mean(y_pred_ann - y)
    r2_ann = r2_score(y, y_pred_ann)
    

    # Display results
    results = {
        "RMSE": round(rmse_ann, 2),
        "NRMSE": round(nrmse_ann * 100, 2),
        "MAPE": round(mape_ann * 100, 2),
        "MBE": round(mbe_ann, 2),
        "R²": round(r2_ann * 100, 2),
    }
    
    print("\nSVM Model Performance:")
    print(f"RMSE: {rmse_ann:.2f}")
    print(f"NRMSE: {nrmse_ann * 100:.2f}%")
    print(f"MAPE: {mape_ann * 100:.2f}%")
    print(f"MBE: {mbe_ann:.2f}")
    print(f"R² (Correlation Coefficient): {r2_ann* 100:.2f}")

    # Save metrics in the database
    ANN.objects.create(
        name="ANN Antenna Structure Prediction",
        rmse=rmse_ann,
        nrmse=nrmse_ann,
        mape=mape_ann,
        mbe=mbe_ann,
       r2_coefficient=r2_ann,
    )


    # Fetch the latest database entry to pass to the template
    data = ANN.objects.last()

    # Save the trained model (optional)
    joblib.dump(ann_model, 'ANN.pkl')
    
    
    return render(req, 'admin/ANN_alg.html', results )


def admingraph(req):
    # Fetch the latest r2_score for each model
    svm_details1 = SVM.objects.last()
    SVM1 = svm_details1.r2_coefficient

    ann_details2 = ANN.objects.last()
    ANN1 = ann_details2.r2_coefficient


    print('SVM1','ANN1')
    print(SVM1,ANN1)
    return render(req, 'admin/admin-graph-analysis.html',{'SVM':SVM1,'ANN':ANN1})