from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_signal, name="upload_signal"),
    path("summary/<str:sid>/", views.summary, name="summary"),
    path("spectrum/<str:sid>/", views.spectrum, name="spectrum"),
    path("wave_previews/<str:sid>/", views.wave_previews, name="wave_previews"),
    path("spectrograms/<str:sid>/", views.spectrograms, name="spectrograms"),

    path("custom_conf/<str:sid>/", views.custom_conf, name="custom_conf"),
    path("equalize/<str:sid>/", views.equalize, name="equalize"),

    path("save_scheme/<str:sid>/", views.save_scheme, name="save_scheme"),
    path("load_scheme/<str:sid>/", views.load_scheme, name="load_scheme"),

    path("save_settings/<str:sid>/", views.save_settings, name="save_settings"),
    path("load_settings/<str:sid>/", views.load_settings, name="load_settings"),

    path("audio/<str:sid>/input.wav", views.audio_input, name="audio_input"),
    path("audio/<str:sid>/output.wav", views.audio_output, name="audio_output"),

    # New AI Endpoint
    path("run_ai/<str:sid>/", views.run_ai, name="run_ai"),

    path("ai_run/<str:sid>/", views.ai_run, name="ai_run"),
]