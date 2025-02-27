from django import forms

class StudentForm(forms.Form):
    name = forms.CharField(label="Name", max_length=100)
    dob = forms.DateField(label="Date of Birth", widget=forms.SelectDateWidget(years=range(1900, 2100)))
    address = forms.CharField(label="Address", widget=forms.Textarea)
    contact_number = forms.CharField(label="Contact Number", max_length=15)
    email = forms.EmailField(label="Email")
    marks_english = forms.IntegerField(label="Marks in English", min_value=0, max_value=100)
    marks_physics = forms.IntegerField(label="Marks in Physics", min_value=0, max_value=100)
    marks_chemistry = forms.IntegerField(label="Marks in Chemistry", min_value=0, max_value=100)
