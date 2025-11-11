# equalizer_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # NO 'api/' prefix here!
    path("upload/", views.upload_signal, name="upload_signal"),
    path("summary/<slug:sid>/", views.summary, name="summary"),
    path("spectrum/<slug:sid>/", views.spectrum, name="spectrum"),
    path("wave_previews/<slug:sid>/", views.wave_previews, name="wave_previews"),
    path("spectrograms/<slug:sid>/", views.spectrograms, name="spectrograms"),
    path("custom_conf/<slug:sid>/", views.custom_conf, name="custom_conf"),

    path("equalize/<slug:sid>/", views.equalize, name="equalize"),

    path("save_scheme/<slug:sid>/", views.save_scheme, name="save_scheme"),
    path("load_scheme/<slug:sid>/", views.load_scheme, name="load_scheme"),
    path("save_settings/<slug:sid>/", views.save_settings, name="save_settings"),
    path("load_settings/<slug:sid>/", views.load_settings, name="load_settings"),

    path("audio/<slug:sid>/input.wav", views.audio_input, name="audio_input"),
    path("audio/<slug:sid>/output.wav", views.audio_output, name="audio_output"),

    path("ai/<slug:sid>/run/", views.ai_run, name="ai_run"),
]