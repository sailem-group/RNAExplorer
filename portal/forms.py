# portal/forms.py
from django import forms

class Bootstrap5FormMixin:
    """Adds BS5 classes and marks invalid fields with is-invalid."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for f in self.fields.values():
            w = f.widget
            if isinstance(w, (forms.TextInput, forms.EmailInput, forms.NumberInput,
                              forms.URLInput, forms.PasswordInput, forms.DateInput)):
                w.attrs["class"] = (w.attrs.get("class","") + " form-control").strip()
            elif isinstance(w, forms.Textarea):
                w.attrs["class"] = (w.attrs.get("class","") + " form-control").strip()
                w.attrs.setdefault("rows", 3)
            elif isinstance(w, forms.Select):
                w.attrs["class"] = (w.attrs.get("class","") + " form-select").strip()
            elif isinstance(w, (forms.RadioSelect, forms.CheckboxSelectMultiple)):
                w.attrs["class"] = (w.attrs.get("class","") + " form-check-input").strip()
            elif isinstance(w, forms.CheckboxInput):
                w.attrs["class"] = (w.attrs.get("class","") + " form-check-input").strip()

        if self.is_bound:
            for name, f in self.fields.items():
                if self.errors.get(name):
                    f.widget.attrs["class"] = f.widget.attrs.get("class","") + " is-invalid"

INPUT_MODE_CHOICES = [
    ("database", "Database"),
    ("text",     "Paste sequences"),
    ("fasta",    "Upload FASTA"),
    ("csv",      "Upload CSV"),
]

FEATURE_CHOICES = [
    ("length",     "Length"),
    ("gc_pct",     "GC %"),
    ("gc_skew",    "GC skew"),
    ("at_au_skew", "AT/AU skew"),
    ("mnc",        "Mononucleotide composition"),
    ("k2",         "k-mer (k=2)"),
    ("k3",         "k-mer (k=3)"),
]

DB_TYPE_CHOICES = [
    ("siRNA", "siRNA"),
    ("miRNA","miRNA"),
    ("piRNA","piRNA"),
]

class FeatureExtractorForm(Bootstrap5FormMixin, forms.Form):

    input_mode = forms.ChoiceField(choices=INPUT_MODE_CHOICES, label="Input source")

    db_types = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=DB_TYPE_CHOICES,
        label="Choose databases",
    )

    # Paste / upload inputs
    sequences_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 8, "placeholder": " >sequenceID-001 description \nAAGTAGGAATAATATCTTATCATTA \n >sequenceID-002 description \nACGACTAGACATATATCAGCTCGC"}),
        label="Sequences",
    )
    file = forms.FileField(required=False, label="Upload file")

    features = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=FEATURE_CHOICES,
        initial=[c for c, _ in FEATURE_CHOICES],
        label="Features",
    )

    def clean(self):
        cd = super().clean()
        mode = cd.get("input_mode")

        if mode == "database":
            if not cd.get("db_types"):
                raise forms.ValidationError("Choose at least one database type.")
        elif mode in ("fasta", "csv"):
            if not cd.get("file"):
                raise forms.ValidationError("Please upload a file.")
        elif mode == "text":
            if not (cd.get("sequences_text") or "").strip():
                raise forms.ValidationError("Please paste sequences.")
        else:
            raise forms.ValidationError("Invalid input source.")
        return cd