from django.urls import path, include
from . import views
from django.views.generic import RedirectView
from core.views import demo_access

urlpatterns = [
    path("accounts/", include("django.contrib.auth.urls")),
    path("demo-access/", demo_access, name="demo_access"),

    path("", RedirectView.as_view(pattern_name="feature_explorer", permanent=True), name="home"),

    # Help
    path("help/", views.help, name="help"),
    path("help/getting-started/", views.help_getting_started, name="help_getting_started"),
    path("about/", views.about, name="about"),

    # Tool
    path("tools/extractor/download/", views.feature_extractor_download, name="feature_extractor_download"),
    path("tools/feature-explorer/download-embeddings/", views.feature_explorer_download_embeddings, name="feature_explorer_download_embeddings"),
    path("feature-explorer/", views.feature_lab, name="feature_explorer"),
]
