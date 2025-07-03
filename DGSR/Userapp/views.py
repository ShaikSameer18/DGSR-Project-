from django.shortcuts import render,redirect
from Mainapp.models import *
from Userapp.models import Feedback,Dataset
from Adminapp.models import *
from django.contrib import messages
import time
from django.core.paginator import Paginator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.core.files.storage import default_storage
from django.conf import settings
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import User, Feedback  # Ensure your models are imported
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def userdashboard(req):
    images_count =  User.objects.all().count()
    print(images_count)
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id)
    return render(req,'user/user-dashboard.html')
  

def profile(req):
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id)
    if req.method == 'POST':
        user_name = req.POST.get('userName')
        user_age = req.POST.get('userAge')
        user_phone = req.POST.get('userPhNum')
        user_email = req.POST.get('userEmail')
        user_address = req.POST.get("userAddress")
        # user_img = request.POST.get("userimg")

        user.Full_name = user_name
        user.Age = user_age
        user.Address = user_address
        user.Phone_Number = user_phone
        user.Email=user_email
       

        if len(req.FILES) != 0:
            image = req.FILES['profilepic']
            user.Image = image
            user.Full_name = user_name
            user.Age = user_age
            user.Address = user_address
            user.Phone_Number = user_phone
            user.Email=user_email
            user.Address=user_address
            
            user.save()
            messages.success(req, 'Updated SUccessfully...!')
        else:
            user.Full_name = user_name
            user.Age = user_age
            user.save()
            messages.success(req, 'Updated SUccessfully...!')
            
    context = {"i":user}
    return render(req, 'user/user-profile.html', context)

def userlogout(req):
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(req, 'You are logged out..')
    return redirect('index')

def userfeedbacks(req):
    id=req.session["User_id"]
    uusser=User.objects.get(User_id=id)
    if req.method == "POST":
        rating=req.POST.get("rating")
        review=req.POST.get("review")
        if not rating:
            messages.info(req,'give rating')
            return redirect('userfeedbacks')
        sid=SentimentIntensityAnalyzer()
        score=sid.polarity_scores(review)
        sentiment=None
        if score['compound']>0 and score['compound']<=0.5:
            sentiment='positive'
        elif score['compound']>=0.5:
            sentiment='very positive'
        elif score['compound']<-0.5:
            sentiment='very negative'
        elif score['compound']<0 and score['compound']>=-0.5:
            sentiment='negative'
        else :
            sentiment='neutral'
        # print(sentiment)        
        # print(rating,feed)
        Feedback.objects.create(Rating=rating, Review=review, Sentiment=sentiment, Reviewer=uusser)
        messages.success(req,'Feedback recorded')
        return redirect('userfeedbacks')
    return render(req,'user/user-feedbacks.html')



import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the ANN model
# ann_model = joblib.load("D:/DGSR/DGSR/ANN.pkl")
import joblib

ann_model = joblib.load('C:/Users/Dell/OneDrive/Desktop/DGSR/DGSR/DGSR/ANN.pkl')
 # Assuming ANN is in SavedModel format
scaler = joblib.load('C:/Users/Dell/OneDrive/Desktop/DGSR/DGSR/DGSR/scaler.pkl')

# If a scaler was used, load it as well (assuming you saved the scaler)
# scaler = joblib.load("D:/DGSR/DGSR/scaler.pkl")

def predict_solar_radiation(request):
    if request.method == 'POST':
        # Collect feature values from form submission
        user_input = {
            'VIS06_Reflectance': float(request.POST['VIS06_Reflectance']),
            'VIS08_Reflectance': float(request.POST['VIS08_Reflectance']),
            'HRV_Reflectance': float(request.POST['HRV_Reflectance']),
            'IR016_Reflectance': float(request.POST['IR016_Reflectance']),
            'Solar_Zenith_Angle_deg': float(request.POST['Solar_Zenith_Angle_deg']),
            'Temperature_C': float(request.POST['Temperature_C']),
            'Humidity_%': float(request.POST['Humidity_%']),
            'Wind_Speed_m/s': float(request.POST['Wind_Speed_m/s']),
        }
        
        # Convert to DataFrame for prediction
        user_df = pd.DataFrame([user_input])
        
        # Apply scaling if the model was trained with standardized data
        # user_df = scaler.transform(user_df)  # Uncomment if scaling was used during training
        
        # Make prediction
        prediction = ann_model.predict(user_df)[0][0]  # Extract the single predicted value
        
        # Render prediction result
        return render(request, 'user/predict_solar_radiation.html', {'prediction': prediction})
    
    # Render the form if request is not POST
    return render(request, 'user/predict_solar_radiation.html')











        





    
   

