from django import forms

CAR_CHOICES = [
    ('toyota', 'Toyota'),
    ('ford', 'Ford'),
    ('bmw', 'BMW'),
    ('honda', 'Honda'),
]

class CarForm(forms.Form):
    manufacturer = forms.ChoiceField(
        choices=CAR_CHOICES,
        label="Car Manufacturer"
    )
    model = forms.CharField(
        max_length=100,
        label="Car Model",
        widget=forms.TextInput(attrs={'placeholder': 'Enter model name'})
    )
