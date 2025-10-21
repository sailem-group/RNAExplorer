# core/middleware.py
from django.shortcuts import redirect
from django.urls import reverse

DEMO_SESSION_KEY = "demo_pass_ok"

class DemoAccessCodeMiddleware:
    """
    Gate the whole site behind a simple access code stored in session.
    Skip static/media/admin/auth URLs so login page and assets work.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path

        # Allow-list (adjust as needed)
        allow = (
            path.startswith("/static/")
            or path.startswith("/media/")
            or path.startswith("/admin/login")
            or path.startswith("/admin/")
            or path.startswith("/accounts/login")
            or path.startswith("/accounts/logout")
            or path.startswith("/healthz")          # if you have probes
        )

        if allow or request.session.get(DEMO_SESSION_KEY, False):
            return self.get_response(request)

        # If not validated, force to access-code page
        from django.urls import reverse
        if path != reverse("demo_access"):
            return redirect("demo_access")

        return self.get_response(request)
