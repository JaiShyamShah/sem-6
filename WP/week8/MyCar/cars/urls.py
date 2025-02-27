# cars/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.car_form_view, name='car_form'),
    path('result/', views.result_view, name='result'),
]
