from django.urls import path
from .views import magazine_cover

urlpatterns = [
    path('', magazine_cover, name='magazine_cover'),
]

