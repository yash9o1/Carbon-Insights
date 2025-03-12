from django.urls import path
from . import views

urlpatterns = [
    path('predict/<str:state>/<str:date>/', views.predict, name='predict'),
]