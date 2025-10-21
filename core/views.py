# core/views.py
from django.conf import settings
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from .middleware import DEMO_SESSION_KEY

@require_http_methods(["GET", "POST"])
def demo_access(request):
    error = None
    if request.method == "POST":
        code = (request.POST.get("code") or "").strip()
        expected = getattr(settings, "DEMO_ACCESS_CODE", "")
        if expected and code == expected:
            request.session[DEMO_SESSION_KEY] = True
            return redirect("/")
        error = "Invalid code"
    return render(request, "demo_access.html", {"error": error})
