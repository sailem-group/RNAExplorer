# portal/templatetags/form_extras.py
from django import template
from django.forms.widgets import (
    RadioSelect, CheckboxInput, CheckboxSelectMultiple, Textarea, Select, SelectMultiple
)

register = template.Library()

@register.filter
def widget_type(bound_field):
    try:
        return bound_field.field.widget.__class__.__name__
    except Exception:
        return ""

@register.filter
def input_type(bound_field):
    return getattr(bound_field.field.widget, "input_type", "") or ""

@register.filter
def is_radio(bound_field):
    return isinstance(bound_field.field.widget, RadioSelect)

@register.filter
def is_checkbox(bound_field):
    return isinstance(bound_field.field.widget, CheckboxInput)

@register.filter
def is_checkbox_multiple(bound_field):
    return isinstance(bound_field.field.widget, CheckboxSelectMultiple)

@register.filter
def is_textarea(bound_field):
    return isinstance(bound_field.field.widget, Textarea)

@register.filter
def is_select(bound_field):
    return isinstance(bound_field.field.widget, Select)

@register.filter
def is_select_multiple(bound_field):
    return isinstance(bound_field.field.widget, SelectMultiple)

@register.filter
def add_attrs(bound_field, arg):
    attrs = {}
    for pair in str(arg).split(","):
        if not pair.strip():
            continue
        k, _, v = pair.partition(":")
        attrs[k.strip()] = v.strip()
    return bound_field.as_widget(attrs=attrs)
