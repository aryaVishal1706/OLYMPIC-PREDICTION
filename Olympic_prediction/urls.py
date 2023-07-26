from django.contrib import admin
from django.urls import path
from Olympic_prediction import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('game/', views.get_sports_types, name='game'),
    path('participant/', views.participant, name='participant'),
    path('country/', views.country, name='country'),
    path('medal/', views.medal, name='medal'),
]
