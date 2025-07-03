from django.db import models

# Create your models here.
class manage_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'manage_users'




class SVM(models.Model):
    # Fields to store results of Support Vector Machine algorithm
    name = models.CharField(max_length=255)  # Name or description of the model/algorithm
    rmse = models.FloatField()                 # Root Mean Squared Error
    nrmse = models.FloatField()                # Normalized Root Mean Squared Error (percentage)
    mape = models.FloatField()                 # Mean Absolute Percentage Error (percentage)
    mbe = models.FloatField()                  # Mean Bias Error
    r2_coefficient = models.FloatField()       # R² (Correlation Coefficient)
    created_at = models.DateTimeField(auto_now_add=True)  # Auto timestamp when results are created

    def __str__(self):
        return f"SVM Results - {self.name}"

    class Meta:
        db_table = 'SVM'  # Set the database table name


class ANN(models.Model):
    # Fields to store results of Artificial Neural Network algorithm
    name = models.CharField(max_length=255)  # Name or description of the model/algorithm
    rmse = models.FloatField()                 # Root Mean Squared Error
    nrmse = models.FloatField()                # Normalized Root Mean Squared Error (percentage)
    mape = models.FloatField()                 # Mean Absolute Percentage Error (percentage)
    mbe = models.FloatField()                  # Mean Bias Error
    r2_coefficient = models.FloatField()       # R² (Correlation Coefficient)
    created_at = models.DateTimeField(auto_now_add=True)  # Auto timestamp when results are created

    def __str__(self):
        return f"ANN Results - {self.name}"

    class Meta:
        db_table = 'ANN'  # Set the database table name




