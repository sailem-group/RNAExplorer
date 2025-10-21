from django.urls import path, include
from . import views
from core.views import demo_access

urlpatterns = [
    path("accounts/", include("django.contrib.auth.urls")),
    path("demo-access/", demo_access, name="demo_access"),

    path("", views.home, name="home"),

    # Help
    path("help/", views.help, name="help"),
    path("help/getting-started/", views.help_getting_started, name="help_getting_started"),
    path("about/", views.about, name="about"),

    # Tools
    path("tools/extractor/", views.feature_extractor, name="feature_extractor"),
    path("tools/extractor/download/", views.feature_extractor_download, name="feature_extractor_download"),

    path("tools/explorer/", views.feature_explorer, name="feature_explorer"),
    path("tools/explorer/results/", views.feature_explorer_results, name="feature_explorer_results"),
    path("tools/feature-explorer/download-embeddings/", views.feature_explorer_download_embeddings, name="feature_explorer_download_embeddings"),
]
