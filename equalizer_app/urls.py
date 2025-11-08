from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('human/', views.human, name='human'),
    path('music/', views.music, name='music'),
    path('animal/', views.animal, name='animal'),
]
