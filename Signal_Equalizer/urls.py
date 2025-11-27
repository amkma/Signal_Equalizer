from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),

    # === Front page (serves templates/index.html) ===
    path("", TemplateView.as_view(template_name="index.html"), name="home"),

    # === All backend/API endpoints stay as they are under /api/ ===
    path("api/", include("equalizer_app.urls")),
]

# serve media/static in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)